flielist="B0950+08_tracking_0001_A.fil B0950+08_tracking_0001_B.fil"
for file in $flielist;
do
    echo "python /home/liu/win/Development/fft_half_precision/python_scripts/plot_bit_data.py --file $file  --channel_num 32768 --fre_add 32 --time_add 4 "
    python /home/liu/win/Development/fft_half_precision/python_scripts/plot_bit_data.py --file $file  --channel_num 32768 --fre_add 32 --time_add 4
done
