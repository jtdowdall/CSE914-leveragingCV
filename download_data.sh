#!/bin/bash
wget http://dl.yf.io/bdd-data/v1/videos/val.zip
unzip val.zip
rm val.zip
mv val/ data/
