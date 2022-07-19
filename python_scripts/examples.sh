#画图的例子
flielist="～/data/B0950+08_tracking_0001_A.fil ～/data/B0950+08_tracking_0001_B.fil"
for file in $flielist;
do
    echo "python plot_bit_data.py --file $file  --channel_num 32768 --fre_add 32 --time_add 4 "
    python plot_bit_data.py --file $file  --channel_num 32768 --fre_add 32 --time_add 4
done

#生成8bit信号的例子

python generate_fft_data.py --file ~/data/20201013_lowfreq_1000_A.dat --fft_length 131072 --freq 40000,40001,40002,40003 --time_interval 200,300,400,500,600,700,600,500,400,300,200,100,200,300,400,500,600,700,600,500,400,300,200 --input_type int8

python generate_fft_data.py --file ~/data/21CMA_16bit_test.dat --fft_length 32768 --freq 10000,10001,10002,10003 --time_interval 100,200,300,100,200,300,50,90,180,332,150,100,72,100,200,300,100,200,300,50,90,180,332,150,100,72,100,200,300,100,200,300,50,90,180,332,150,100,72,100,200,300,100,200,300,50,90,180,332,150,100,72 --input_type int16
