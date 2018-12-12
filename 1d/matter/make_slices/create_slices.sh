END=5000
for ((i=0;i<=END;i++)); do
    ./ex10 -m 5000 -slice $i
done
