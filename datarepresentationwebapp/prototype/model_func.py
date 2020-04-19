import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .helper_func import hist_custom, normal_custom


# model functions
def read_input_file(filename):
    """df : mean, std"""
    # from file
    df = pd.read_excel(filename)
    return df


def read_input_user():
    sample_number = int(input("sample number"))
    return sample_number


def run_f(df, sample_number):
    """return sampled array from the distribution in the model"""
    samples = normal_custom(df.get(Model.MEAN_KEY), df.get(Model.STD_KEY), n_sample=sample_number)  # Normal_custom imported from helper_func
    return samples


class Model:
    MEAN_KEY = "mean"
    STD_KEY = "std"

    @classmethod
    def run(cls, df, sample_number):
        """run model"""
        samples = run_f(df, sample_number)
        fig1, ax1 = plt.subplots()
        hist_custom(samples, ax=ax1)
        print('solved, mean={},std={}:'.format(df.get(cls.MEAN_KEY), df.get(cls.STD_KEY)))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.show()
        return df, buf

    @classmethod
    def calibrate(cls, df, sample_number):
        new_std = 10 * df.get(cls.STD_KEY)
        df["std"] = new_std
        fig2, ax2 = plt.subplots()
        hist_custom(run_f(df, sample_number), ax=ax2)
        print('calibrated mean={},std={}:'.format(df.get(cls.MEAN_KEY), df.get(cls.STD_KEY)))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.show()
        return df, buf

    @classmethod
    def report(cls):
        """Fake report function implementation"""
        x = np.random.randint(low=1, high=11, size=50)
        y = x + np.random.randint(1, 5, size=x.size)
        fig, ax1 = plt.subplots()
        ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
        ax1.set_title('Scatter: $x$ versus $y$')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.show()
        return buf


if __name__ == "__main__":
    Model.report()
    

    

