import pandas as pd
import numpy as np
import sys

# Number of rows
num_rows = 10
if len(sys.argv) > 1:
    try:
        num_rows = int(sys.argv[1])
    except:
        pass

# Column names exactly matching the training data
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Categorical values
protocol_types = ["tcp", "udp", "icmp"]
services = ["http", "ftp", "smtp", "domain_u"]
flags = ["SF", "S0", "REJ"]

# Prepare data dictionary
data_dict = {}

for col in columns:
    if col == "protocol_type":
        data_dict[col] = np.random.choice(protocol_types, num_rows)
    elif col == "service":
        data_dict[col] = np.random.choice(services, num_rows)
    elif col == "flag":
        data_dict[col] = np.random.choice(flags, num_rows)
    else:
        # numeric columns
        data_dict[col] = np.random.randint(0, 100, num_rows)

# Create DataFrame
df = pd.DataFrame(data_dict)

# Output file
filename = "large_test_input_realistic.csv" if num_rows > 50 else "test_input_realistic.csv"

df.to_csv(filename, index=False)
print(f"✅ {filename} created with {num_rows} rows and {len(columns)} columns")
