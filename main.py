import numpy as np
import pandas as pd
from aws_to_df import AwsToDf


def run_program(name):

    atd = AwsToDf()
    df = atd.sql_to_df('test_sql')
    print(df.head())


if __name__ == '__main__':
    run_program('PyCharm')
