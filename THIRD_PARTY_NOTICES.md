# Third-Party Notices

WeSep is distributed under the Apache License 2.0 except where noted below.
The source comments identify the corresponding implementations in more detail.

## ESPnet TFGridNet

`wesep/modules/separator/tfgridnet.py` is based on ESPnet's TFGridNetV2
implementation, licensed under Apache License 2.0. WeSep reorganizes the
separator and adds modular cue interfaces, multi-output support, and causal
options.

Source: https://github.com/espnet/espnet

## DAE-TSE KCE

`wesep/modules/textual/kce.py` and the textual TSE integration follow the
official DAE-TSE implementation, licensed under Apache License 2.0. The code
has been reorganized for WeSep's frontend and deferred-loading interfaces.

Source: https://github.com/GnafiY/DAE-TSE

## MuSE Visual Frontend

`wesep/modules/visual/muse.py` follows the MuSE visual frontend. The upstream
repository does not publish a license file; its author has granted permission
for this WeSep adaptation to be released under Apache License 2.0.

Source: https://github.com/zexupan/MuSE

## USEF-TSE

`wesep/modules/speaker/usef.py` is based on USEF-TSE, whose upstream repository
is licensed under Creative Commons Attribution-NonCommercial 4.0 International.
This file is excluded from WeSep's Apache License 2.0 grant and remains subject
to the upstream non-commercial license.

Source: https://github.com/ZBang/USEF-TSE
License: https://creativecommons.org/licenses/by-nc/4.0/

## NBSS NBC2

`wesep/modules/separator/nbc2.py` is adapted from the official NBSS narrow-band
implementation and remains subject to the following MIT License notice.

Copyright (c) 2024 Changsheng Quan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
