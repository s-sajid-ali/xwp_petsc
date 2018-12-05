END=100
for ((i=1;i<=END;i++)); do
    ./ex10 -m 512 -slice $i
done
