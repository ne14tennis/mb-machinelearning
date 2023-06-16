import os
import boto3
import pandas as pd
from typing import Optional
from newtools import AthenaClient, PandasDoggo, S3Location


class AwsToDf:

    def __init__(self, bucket_name=None):
        self.bucket_name = bucket_name if not None else 'csmediabrain-mediabrain'
        self.bucket_location = 's3://' + self.bucket_name + '/'
        self.bucket = S3Location(self.bucket_location)
        self.doggo = PandasDoggo()
        self.ac = AthenaClient('us-west-2', 'mediabrain', workgroup='primary')
        self.output_location = self.bucket.join('aws-athena-query-results-us-west-2')

    def sql_to_df(self, sql_name: str) -> pd.DataFrame:
        """
        Gets the result of a local sql file as a dataframe

        :param sql_name: name of the sql file (accepts with or without the .sql on the end)
        :return: sql result
        """

        sql_file = sql_name if sql_name.endswith('.sql') else sql_name + '.sql'

        sql = os.path.abspath(os.path.join(os.getcwd(), sql_file))
        with open(sql) as sql_script:
            query = sql_script.read()
            qr = self.ac.add_query(query, output_location=self.output_location)

        print(f'Running query {sql_file}')
        self.ac.wait_for_completion()
        result = self.ac.get_query_result(qr)
        print(f'Result obtained for {sql_file}')

        return result

    def files_to_df(self,
                    file_prefix: str,
                    file_key: Optional[str] = None,
                    file_type: Optional[str] = None,
                    has_header: Optional[bool] = False
                    ) -> pd.DataFrame:
        """
        Loads all files, or a single file, as a single dataframe

        :param file_prefix: prefix of the folder, without the bucket
        (e.g. 'prod_mb/example_folder/', rather than 's3://csmediabrain-mediabrain/prod_mb/example_folder/')
        :param file_key: if only wanting a single file, the name of the file itself
        :param file_type: allowed types are 'csv' and 'parquet', leave blank for inferred type
        :param has_header: whether the file has a header as the first row
        :return: all files as a single dataframe
        """

        file_prefix = file_prefix if file_prefix.endswith('/') else file_prefix + '/'
        if file_key is not None:
            file_key = file_key if file_key.endswith('.' + file_type) else file_key + '.' + file_type

        boto_session = boto3.Session()
        s3 = boto_session.resource('s3')
        boto_bucket = s3.Bucket(self.bucket_name)

        if file_key:
            file_location = S3Location(self.bucket_location + file_prefix + file_key)
            full_df = self._load_s3_file(file_location, file_type, has_header)
        else:
            files = []
            for file in boto_bucket.objects.filter(Prefix=file_prefix):
                file_location = self.bucket_location + file.key
                df_temp = self._load_s3_file(file_location, file_type, has_header)
                files.append(df_temp)
                if len(files) % 100 == 0:
                    print(f'{len(files)} files loaded so far')

            print('Finished loading all files')
            full_df = pd.concat(files, ignore_index=True)

        return full_df

    def _load_s3_file(self, full_file_path, file_type, has_header):

        if file_type == 'csv':
            if has_header:
                df = self.doggo.load_csv(full_file_path)
            else:
                df = self.doggo.load_csv(full_file_path, header=None)
        elif file_type == 'parquet':
            if has_header:
                df = self.doggo.load_parquet(full_file_path)
            else:
                df = self.doggo.load_parquet(full_file_path, header=None)
        else:
            if has_header:
                df = self.doggo.load(full_file_path)
            else:
                df = self.doggo.load(full_file_path, header=None)

        return df
