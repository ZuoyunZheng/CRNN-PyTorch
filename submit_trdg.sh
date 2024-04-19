rm data/trdg10 -r

parts=$(($1 / 10000))
echo "Splitting into $parts parts a la 10k samples"
for (( p=1; p<=$parts; p++ )); do
    echo "$p/$parts"
    source trdg.sh --output_dir data/trdg10/horizontal/lvis/$p -c 10000 -or 0 -b 3
done
