from sec_api import QueryApi
import json
import pprint

queryApi = QueryApi(api_key="a41750122ec15749131a56ce87ab6b23bd458c1c0328f99151241f185904a7d8")

query = {
  "query": {"query_string":
    {
      "query": "ticker:TSLA AND filedAt:{2020-01-01 TO 2020-12-31} AND formType:\"10-Q\""
    } },
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
}

filings = queryApi.get_filings(query)
json_data = json.dumps(filings)
pprint.pprint(json_data)


