# vi-ba-nmt-api

## Setup
**Install environment:** <br/>
$ conda env create -f environment.yml <br/>
$ conda activate vi-ba-nmt-api <br/>
**Prepare vncorenlp service:** <br/>
$ cd transformerbertpgn <br/>
$ ./prepare_vncorenlp.sh <br/>
**Get models' weights:** <br/>
Get models' weights from: https://drive.google.com/drive/folders/1JVtzatiNdcFFaODza8Hhr4aMBo7OFYhe?usp=sharing <br/>
Extract bert-fused/tbmp's weights: <br/>
$ cd transformerbertpgn <br/>
$ tar -xzvf transformerbertpgn.tar.gz <br/>
Extract transformer's weights: <br/>
$cd transformer_based <br/>
$ tar -xzvf transformer-based.tar.gz <br/>


## Run
**Run vncorenlp service:** <br/>
$ cd transformerbertpgn <br/>
$ ./vncorenlp_service.sh <br/>
**Start server:** <br/>
$ python run.py

## References
Bùi Ngô Hoàng Long. https://github.com/buingohoanglong/transformer-bert-pgn <br/>
Nguyễn Sư Phước. https://github.com/nsphuoc10/transformer-based
