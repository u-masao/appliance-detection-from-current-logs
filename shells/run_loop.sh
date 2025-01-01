#!/bin/bash

uv sync -q

make sync_from_storage

while true
do
  make train sync_to_storage
done
