SELECT series_no, name
FROM mediabrain.series
WHERE batch_id = (SELECT MAX(batch_id) FROM mediabrain.series)
AND linear_db_schema = 'linear_national'



