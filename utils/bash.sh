#!/bin/bash
#anteriores
for base in \
  "eb0203196e284797_1125" \
  "eb0203196e284797_1126" \
  "eb0203196e284797_1130" \
  "eb0203196e284797_1131" \
  "eb0203196e284797_1135" \
  "eb0203196e284797_1136" \
  "eb0203196e284797_1137" \
  "fb86bc87d3874cd7_3660"
do
  find . -type f -name "${base}.*" -print
done

#nuevas

for base in \
  "eb0203196e284797_1125" \
  "eb0203196e284797_1126" \
  "eb0203196e284797_1130" \
  "eb0203196e284797_1131" \
  "eb0203196e284797_1135" \
  "eb0203196e284797_1136" \
  "eb0203196e284797_1137" \
  "fb86bc87d3874cd7_3660"
do
  echo "===== $base ====="
  find . -type f -name "${base}_1.*" -print
done