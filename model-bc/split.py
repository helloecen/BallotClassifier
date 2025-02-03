import splitfolders

data_in = 'data1'
data_out = 'data1_split'

splitfolders.ratio(input=data_in,
                   output=data_out,
                   ratio=(0.8, 0.2))