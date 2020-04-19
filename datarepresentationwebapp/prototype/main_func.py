from .model_func import *

if __name__ == "__main__":
    # read data
    sample_number = read_input_user()
    df = read_input_file('data.xlsx')

    # Initialize model
    model = Model(df, sample_number)
    # run model
    model.run()
    # calibrate
    model_cal = model.calibrate()
    # re-run
    model_cal.run()
