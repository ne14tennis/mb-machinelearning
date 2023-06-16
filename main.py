import numpy as np
import pandas as pd
from aws_to_df import AwsToDf


def run_program(name):

    atd = AwsToDf()
    df = atd.sql_to_df('test_sql')
    print(df.head())
    print('We are in test 1')


if __name__ == '__main__':
    run_program('PyCharm')
