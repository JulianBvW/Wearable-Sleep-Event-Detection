from torchinfo import summary

def show_script_info(args, model, input_channels):
    print('--=={ Running script with arguments }==--')
    for k, v in vars(args).items():
        print(f'    {k:17}: {v}')
    print('--=====================================--')
    model_info = str(summary(model, input_size=(args.multi_batch_size * args.batch_size, input_channels, args.seq_length), verbose=0))
    model_info = model_info.split('==========================================================================================')
    param_info, input_info = model_info[3].split('\n')[1:5], model_info[4].split('\n')[1:5]
    for info in param_info:
        print('   ', info)
    print('--=====================================--')
    for info in input_info:
        print('   ', info)
    print('--=====================================--')