# Data Collection Scripts

Scripts for fetching Bitcoin transaction address data from public APIs.

## fetch_blockchain_api.py

Fetches transaction details (sender/receiver addresses, values, timestamps) using the free Blockchain.com API.

### Features
- ✅ Rate limiting (50 requests/minute)
- ✅ Automatic checkpointing (resume from failures)
- ✅ Progress tracking
- ✅ Error handling and retry logic
- ✅ CSV output with transaction addresses

### Usage

#### Basic Usage
```bash
python src/data_collection/fetch_blockchain_api.py
```

**That's it!** The script automatically:
- Reads from `raw_data/Result.csv`
- Creates `processed_data/api_outputs.csv`
- Creates `processed_data/` directory if needed

#### With Custom Rate Limit
```bash
python src/data_collection/fetch_blockchain_api.py --rate-limit 40
```

#### With Custom Checkpoint Interval
```bash
python src/data_collection/fetch_blockchain_api.py --checkpoint-interval 50
```

### Input Format

CSV file with `txId` and `transaction` columns (Result.csv format):

```csv
txId,transaction
230325127,d6176384de4c0b98702eccb97f3ad6670bc8410d9da715fe5b49462d3e603993
230325139,300c7e7bb34263eae7ff8b0a726d5869bf73d71081490c45a9536a31560f1fd7
86875675,7c790a31090462d720a172b3f55a51af2514971070db6686e337ccc486840dcd
...
```

This maps Elliptic's anonymized transaction IDs to real Bitcoin transaction hashes.

### Output Format

CSV file with columns (txId and transaction first for easy joining):

- `txId`: Elliptic transaction ID
- `transaction`: Bitcoin transaction hash
- `success`: Boolean (True if fetch succeeded)
- `timestamp`: Unix timestamp
- `block_height`: Block number
- `num_inputs`: Number of input addresses
- `num_outputs`: Number of output addresses
- `input_addresses`: Pipe-separated list of sender addresses
- `output_addresses`: Pipe-separated list of receiver addresses
- `input_values`: Pipe-separated values in satoshis
- `output_values`: Pipe-separated values in satoshis
- `total_input_btc`: Total input in BTC
- `total_output_btc`: Total output in BTC
- `fee_btc`: Transaction fee in BTC
- `fetched_at`: ISO timestamp of fetch
- `error`: Error message (if failed)

### Checkpointing

Progress is automatically saved every 100 transactions (configurable):
```bash
python fetch_blockchain_api.py \
    --input tx_hashes.txt \
    --output results.csv \
    --checkpoint-interval 500
```

If interrupted, re-run the same command to resume from checkpoint.

### Rate Limiting

Default: 50 requests/minute (safe for Blockchain.com API)

The API enforces ~60 requests/minute, but 50 provides a safety buffer.

**Time estimates:**
- 1,000 transactions: ~20 minutes
- 10,000 transactions: ~3.3 hours
- 100,000 transactions: ~33 hours
- 203,769 transactions: ~68 hours (~2.8 days)

### Example: Complete Workflow

```bash
# Navigate to project directory
cd /c/Github/BT4012_elliptic

# Run fetcher (reads Result.csv, outputs to api_outputs.csv)
python src/data_collection/fetch_blockchain_api.py

# Estimated time: ~68 hours (~2.8 days) for all 202,804 transactions

# If interrupted, just re-run the same command to resume:
python src/data_collection/fetch_blockchain_api.py

# Monitor progress (in another terminal):
python -c "
import pandas as pd
try:
    df = pd.read_csv('processed_data/api_outputs.checkpoint.csv')
    print(f'Progress: {len(df):,} / 202,804 ({len(df)/202804*100:.1f}%)')
    print(f'Success rate: {df[\"success\"].sum() / len(df) * 100:.1f}%')
except:
    print('No checkpoint yet')
"

# After completion, analyze results:
python -c "
import pandas as pd
df = pd.read_csv('processed_data/api_outputs.csv')
print(f'Total: {len(df):,}')
print(f'Successful: {df[\"success\"].sum():,} ({df[\"success\"].sum()/len(df)*100:.1f}%)')
print(f'Failed: {(~df[\"success\"]).sum():,}')
"
```

### Error Handling

Common errors:
- `Transaction not found (404)`: Hash doesn't exist on blockchain
- `HTTP Error`: API issues (auto-retries 3 times)
- `Rate limited`: Automatic exponential backoff

### Notes

✅ **You have real Bitcoin transaction hashes in `raw_data/Result.csv`!**

The Result.csv file contains a mapping from Elliptic's anonymized txIds to real Bitcoin transaction hashes. This allows you to:
1. Fetch real sender/receiver addresses from the blockchain
2. Build address-level fraud detection models
3. Enrich the Elliptic dataset with address features
4. Perform both transaction-level AND address-level analysis

### Performance Tips

**Speed up with multiple IPs:**
```bash
# Terminal 1 (home network)
python fetch_blockchain_api.py --input batch1.txt --output results1.csv

# Terminal 2 (school network, different IP)
python fetch_blockchain_api.py --input batch2.txt --output results2.csv

# Merge results later
python -c "
import pandas as pd
df1 = pd.read_csv('results1.csv')
df2 = pd.read_csv('results2.csv')
pd.concat([df1, df2]).to_csv('combined.csv', index=False)
"
```

### API Documentation

Blockchain.com API: https://www.blockchain.com/explorer/api/blockchain_api
