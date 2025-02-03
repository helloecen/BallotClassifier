import splitfolders

data_in = 'data2'
data_out = 'data2_split'

splitfolders.ratio(input=data_in,
                   output=data_out,
                   ratio=(0.8, 0.2))
