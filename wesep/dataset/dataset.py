# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Shuai Wang (wsstriving@gmail.com)
#               2026 Ke Zhang (kylezhang1118@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wesep.dataset.processor as processor
from wesep.dataset.cues import build_cue_layer
from wesep.utils.file_utils import load_json, load_yaml, read_lists


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """Return an iterator over the source dataset processed by the
        given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f, *args, **kw):
        if args or kw:
            return Processor(self, f, *args, **kw)
        else:
            raise ValueError("Processor.apply() requires explicit args/kw. "
                             "Implicit parameter inheritance is forbidden.")


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """Sample data according to rank/world_size/num_workers

        Args:
            data(List): input data list

        Returns:
            List: data list after sample
        """
        data = list(range(len(data)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(data)
        if self.partition:
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self,
                 lists,
                 shuffle=True,
                 partition=True,
                 repeat_dataset=False,
                 fallback_on_empty=False):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.fallback_on_empty = fallback_on_empty
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)

        # Keep every training worker active when shards are scarce.
        if not indexes and self.fallback_on_empty and self.lists:
            seed = (f"{self.sampler.epoch}:"
                    f"{sampler_info['rank']}:"
                    f"{sampler_info['worker_id']}")
            indexes = [random.Random(seed).randrange(len(self.lists))]
        if not indexes:
            return

        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            while True:
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def Dataset(
    data_type,
    data_list_file,
    configs,
    state="train",
    repeat_dataset=False,
    cues_yaml=None,
    expand_targets=False,
):
    assert data_type in ["shard", "raw"]

    lists = read_lists(data_list_file)
    shuffle = configs.get("shuffle", False)
    # Online mixing builds random training mixtures; evaluation uses fixed data.
    online_mix = configs.get("online_mix", False) and state == "train"

    dataset = DataList(
        lists,
        shuffle=shuffle,
        repeat_dataset=repeat_dataset,
        fallback_on_empty=state == "train" and data_type == "shard",
    )

    # 1) Source layer
    source_len_policy = "strict"
    dataset = build_source_layer(dataset, data_type, online_mix,
                                 source_len_policy)

    # 2) Basic audio preprocessing
    dataset = build_audio_base_layer(dataset, configs, state, online_mix)

    # 3) Online mix & augmentation
    if state == "train":
        dataset = build_mix_layer(dataset, configs, online_mix)

    # 4) Cue layer (speaker / visual / spatial / semantic)
    if cues_yaml is not None:
        dataset = build_cue_layer(dataset, cues_yaml, state, configs)

    # 5) Optional target-level view for one-speaker-at-a-time inference.
    if expand_targets:
        dataset = Processor(dataset, processor.expand_target_samples)

    if cues_yaml is not None:
        labels = load_yaml(cues_yaml).get("labels", {})
        if "speaker_label" in labels:
            dataset = Processor(
                dataset,
                processor.add_speaker_labels,
                load_json(labels["speaker_label"]["resource"]),
            )

    return dataset


def build_source_layer(dataset,
                       data_type,
                       online_mix,
                       source_len_policy="strict"):
    # 1) Source layer
    if data_type == "shard":
        dataset = Processor(dataset, processor.url_opener)
        if not online_mix:
            dataset = Processor(dataset, processor.tar_file_and_group,
                                source_len_policy)
        else:
            dataset = Processor(dataset,
                                processor.tar_file_and_group_single_spk)
    else:
        if not online_mix:
            dataset = Processor(dataset, processor.parse_raw,
                                source_len_policy)
        else:
            dataset = Processor(dataset, processor.parse_raw_single_spk)
    return dataset


def build_audio_base_layer(dataset, configs, state, online_mix):
    # 2) Basic audio preprocessing
    if state == "train":
        if configs.get("shuffle", False) and not online_mix:
            dataset = Processor(
                dataset,
                processor.shuffle,
                **configs["shuffle_args"],
            )

    resample_rate = configs.get("resample_rate", 16000)
    dataset = Processor(dataset, processor.resample, resample_rate)

    whole_utt = configs.get("whole_utt", False)
    if state == "train":
        if whole_utt:
            # Whole-utterance training keeps waveform timing intact; the
            # optional length filter only drops extreme samples.
            if configs.get("filter_len", False):
                filter_conf = configs.get("filter_len_args", {})
                dataset = Processor(dataset, processor.filter_len,
                                    **filter_conf)
        else:
            chunk_len = configs.get("chunk_len", resample_rate * 3)
            chunk_random_start = configs.get("chunk_random_start", True)
            chunk_short_policy = configs.get("chunk_short_policy", "pad")
            dataset = Processor(
                dataset,
                processor.random_chunk,
                chunk_len,
                random_start=chunk_random_start,
                short_policy=chunk_short_policy,
            )

    return dataset


def build_mix_layer(dataset, configs, online_mix):
    # 3) Online mix & augmentation
    if online_mix:
        whole_utt = configs.get("whole_utt", False)
        timeline_conf = None if whole_utt else configs.get("timeline", None)

        dataset = Processor(
            dataset,
            processor.sample_speaker_group,
            configs.get("num_speakers", None),
            configs.get("online_buffer_size", 1000),
            timeline_conf,
            whole_utt,
        )
        dataset = Processor(dataset, processor.apply_timeline)

        # This stage can later host shared-room multi-channel rendering.
        dataset = Processor(
            dataset,
            processor.add_reverb,
            configs.get("reverb_prob", 0),
            configs.get("reverb_conf", None),
        )
        dataset = Processor(
            dataset,
            processor.snr_mixer,
            configs.get("mix_snr", None),
        )

    if configs.get("noise_prob", 0) > 0:
        dataset = Processor(
            dataset,
            processor.add_noise,
            configs.get("noise_lmdb_file"),
            configs.get("noise_prob"),
        )

    return dataset
