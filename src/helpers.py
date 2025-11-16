from dataclasses import dataclass, field
from io import BytesIO
import os
from typing import Optional, List, Dict, Any

import sqlalchemy
from azure.identity import AzureCliCredential
import struct
import pyodbc
import yaml
import polars as pl
import duckdb
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class DataLakeManager:
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    service_client: Optional[DataLakeServiceClient] = field(init=False, default=None)

    def __post_init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Load credentials from environment variables if not provided
        if not self.account_name:
            self.account_name = os.getenv("DATALAKE_ACCOUNT")
        if not self.account_key:
            self.account_key = os.getenv("DATALAKE_KEY")

        if not self.account_name or not self.account_key:
            raise ValueError(
                "DATALAKE_ACCOUNT and DATALAKE_KEY must be provided either as "
                "constructor arguments or as environment variables."
            )

        self.service_client = self._get_datalake_client()

    def connect(self, db="replica") -> sqlalchemy.engine.base.Connection:
        match db:
            case "analytics":
                server = "empoweranalyticssql.database.windows.net"
                database = "empower-analytics"
            case "prod" | "replica":
                server = "efscuprodsqlfog01.secondary.database.windows.net"
                database = "sqldb-empower-latam-prod"
            case "us":
                server = "sql-empower-prod-uscentral.database.windows.net"
                database = "empower-prod2"
            case "mx-analytics":
                server = "empower-synapse-latam-prod-ondemand.sql.azuresynapse.net"
                database = "master"
            case "petal-tmp":
                server = "efwusdssql01.database.windows.net"
                database = "efwusdsdb01"
            case "MxLakehouse" | "MxWarehouse":
                server = "7mckfbctd25u7hka3rvsikrvfm-fslebbtya7furbynx2og223n34.datawarehouse.fabric.microsoft.com"
                database = db
            case _:
                raise ValueError(f"Unknown database: {db}")

        credential = AzureCliCredential()
        database_token = credential.get_token("https://database.windows.net/")

        token_bytes = bytes(database_token.token, "UTF-8")
        expanded_token = b""
        for i in token_bytes:
            expanded_token += bytes({i})
            expanded_token += bytes(1)
        token_struct = struct.pack("=i", len(expanded_token)) + expanded_token

        conn_string = (
            f"Driver={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            "ColumnEncryption=Enabled;ColumnEncryptionSetting=Enabled;"
        )
        if database:
            conn_string += f"DATABASE={database};"

        if db == "replica":
            conn_string += "ApplicationIntent=ReadOnly;"

        SQL_COPT_SS_ACCESS_TOKEN = 1256
        conn = pyodbc.connect(
            conn_string,
            attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct},
        )

        return conn

    def _get_datalake_client(self) -> Optional[DataLakeServiceClient]:
        if not self.account_name or not self.account_key:
            raise ValueError("Both account_name and account_key must be provided")

        try:
            service_client = DataLakeServiceClient(
                account_url=f"https://{self.account_name}.dfs.core.windows.net",
                credential=self.account_key
            )
            return service_client
        except Exception as e:
            print(f"Failed to create DataLakeServiceClient: {e}")
            return None



@dataclass
class TableManager(DataLakeManager):
    """
    Extends DataLakeManager with table and directory management capabilities.
    Inherits all DataLake connection and authentication functionality from DataLakeManager.
    """

    container_name: Optional[str] = None
    directory_path: Optional[str] = None
    tables: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False)
    contents: Dict[str, List[str]] = field(default_factory=dict, init=False)
    last_listed_path: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        # Call parent's __post_init__ to initialize DataLake connection
        super().__post_init__()

        # Automatically populate tables if container_name and directory_path are provided
        if self.container_name and self.directory_path:
            self.get_tables()

    def get_tables(self, directory_path: Optional[str] = None, print_contents: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Retrieves all tables from the specified directory and stores them in the tables property.

        Args:
            directory_path: Optional path to the directory containing tables.
                          If not provided, uses self.directory_path.
            print_contents: Whether to print the directory contents. Default is False.

        Returns:
            Dictionary mapping table names to their metadata (type and directory)
            Format: {
                'table_name': {
                    'type': 'incremental' or 'full',
                    'directory': 'path/to/directory' or None (for full tables)
                }
            }

        Raises:
            ValueError: If directory_path is not provided and self.directory_path is not set
        """
        # Use instance directory_path if not provided
        if directory_path is None:
            if not self.directory_path:
                raise ValueError("directory_path must be provided either as argument or during initialization")
            directory_path = self.directory_path

        self.list_directory_contents(directory_path)

        # Print contents if requested
        if print_contents:
            self.print_directory_contents(directory_path)

        self.tables = {}

        # Add incremental tables (directories)
        for directory in self.contents['directories']:
            self.tables[directory] = {
                'type': 'incremental',
                'directory': f"{directory_path}/{directory}"
            }

        # Add full tables (files)
        for file in self.contents['files']:
            # Remove .parquet extension if present
            table_name = file.replace('.parquet', '')
            self.tables[table_name] = {
                'type': 'full',
                'directory': f"{directory_path}/{file}"
            }

        return self.tables

    def list_directory_contents(self, directory_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Lists only the immediate files and folders in a specified directory in Azure Data Lake.
        Does not recurse into subdirectories.

        Args:
            directory_path: Optional path to the directory within the container (e.g., 'data-tmp/ensenada/raw').
                          If not provided, uses self.directory_path.

        Returns:
            Dictionary with 'files' and 'directories' keys, each containing a list of names (not full paths)

        Raises:
            RuntimeError: If DataLake service client is not initialized
            ValueError: If container_name is not set in instance, or if directory_path is not provided
                       and self.directory_path is not set
            ResourceNotFoundError: If the specified path doesn't exist
        """
        if not self.service_client:
            raise RuntimeError("DataLake service client is not initialized")

        if not self.container_name:
            raise ValueError("container_name must be provided during initialization")

        # Use instance directory_path if not provided
        if directory_path is None:
            if not self.directory_path:
                raise ValueError("directory_path must be provided either as argument or during initialization")
            directory_path = self.directory_path

        # Get the file system (container) client
        file_system_client = self.service_client.get_file_system_client(self.container_name)

        # Normalize the directory path (remove trailing slash if present)
        directory_path = directory_path.rstrip('/')

        # List all paths in the directory (non-recursive)
        files = []
        directories = []

        try:
            # Get paths with recursive=False to only get immediate children
            paths = file_system_client.get_paths(path=directory_path, recursive=False)

            for path in paths:
                path_name = path.name

                # Skip the directory itself
                if path_name == directory_path:
                    continue

                # Extract just the name (not the full path)
                # Remove the directory_path prefix to get only the immediate child name
                if path_name.startswith(directory_path + '/'):
                    child_name = path_name[len(directory_path) + 1:]

                    # Only include if it's a direct child (no additional slashes)
                    if '/' not in child_name:
                        if path.is_directory:
                            directories.append(child_name)
                        else:
                            files.append(child_name)

        except ResourceNotFoundError:
            print(f"Path not found: {self.container_name}/{directory_path}")
            print("The directory may not exist or may be empty.")
        except Exception as e:
            print(f"Error listing contents: {e}")
            raise

        # Store contents and path in instance variables
        self.contents = {
            'files': sorted(files),
            'directories': sorted(directories)
        }
        self.last_listed_path = directory_path

        return self.contents

    def print_directory_contents(self, directory_path: Optional[str] = None) -> None:
        """
        Prints a formatted display of directory contents stored in self.contents.

        Args:
            directory_path: Optional path to display in the header. If not provided, uses self.last_listed_path,
                          then falls back to self.directory_path.

        Raises:
            ValueError: If self.contents is empty (call list_directory_contents() first)
        """
        if not self.contents:
            raise ValueError("No contents to print. Call list_directory_contents() first.")

        # Use last_listed_path, then instance directory_path if not provided
        if directory_path is None:
            directory_path = self.last_listed_path or self.directory_path

        files = self.contents.get('files', [])
        directories = self.contents.get('directories', [])

        print(f"\n=== Immediate contents of {self.container_name}/{directory_path} ===")
        print(f"\nIncremental tables ({len(directories)}):")
        for directory in directories:
            print(f"  📁 {directory}")

        print(f"\nFull tables ({len(files)}):")
        for file in files:
            print(f"  📄 {file.replace('.parquet', '')}")

        print(f"\nTotal: {len(directories)} directories, {len(files)} files (immediate children only)")

    def read_table(self, table_name: str, date_str: Optional[str] = None) -> pl.DataFrame:
        """
        Reads a table from Azure Data Lake into a Polars DataFrame.

        Args:
            table_name: Name of the table (key from self.tables)
            date_str: Required for incremental tables. Date string in format 'YYYY-MM-DD'
                     to specify which parquet file to read (e.g., '2025-05-01')

        Returns:
            Polars DataFrame containing the table data

        Raises:
            ValueError: If table_name not found in self.tables, or if date_str is missing for incremental tables,
                       or if container_name is not set in instance
            RuntimeError: If DataLake service client is not initialized
            FileNotFoundError: If the specified parquet file doesn't exist
        """
        if not self.service_client:
            raise RuntimeError("DataLake service client is not initialized")

        if not self.container_name:
            raise ValueError("container_name must be provided during initialization")

        if table_name not in self.tables:
            raise ValueError(
                f"Table '{table_name}' not found in tables. "
                f"Available tables: {list(self.tables.keys())}"
            )

        table_info = self.tables[table_name]
        table_type = table_info['type']
        base_directory = table_info['directory']

        # Determine the full path to the parquet file
        if table_type == 'incremental':
            if not date_str:
                raise ValueError(
                    f"Table '{table_name}' is an incremental table. "
                    f"Please provide a date_str parameter (e.g., '2025-05-01')"
                )
            # For incremental tables: directory/date.parquet
            parquet_path = f"{base_directory}/{date_str}.parquet"
        else:
            # For full tables: the directory already points to the .parquet file
            parquet_path = base_directory

        # Get the file system client
        file_system_client = self.service_client.get_file_system_client(self.container_name)

        try:
            # Get the file client
            file_client = file_system_client.get_file_client(parquet_path)

            # Download the file content
            download = file_client.download_file()
            file_content = download.readall()

            # Read into Polars DataFrame
            df = pl.read_parquet(BytesIO(file_content))

            print(f"✓ Successfully read table '{table_name}' from {self.container_name}/{parquet_path}")
            print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

            return df

        except ResourceNotFoundError:
            raise FileNotFoundError(
                f"Parquet file not found: {self.container_name}/{parquet_path}"
            )
        except Exception as e:
            print(f"Error reading table '{table_name}': {e}")
            raise

    def _get_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Creates and configures a DuckDB connection with Azure credentials.

        Returns:
            Configured DuckDB connection with Azure secret

        Raises:
            RuntimeError: If account credentials are not available
        """
        if not self.account_name or not self.account_key:
            raise RuntimeError(
                "Azure storage account credentials are required for DuckDB queries. "
                "Please provide account_name and account_key when initializing TableManager."
            )

        # Set environment variables that the underlying CURL library uses BEFORE creating connection
        import ssl
        import os

        cert_paths = [
            '/etc/ssl/certs/ca-certificates.crt',   # Ubuntu/Debian system certs (try first)
            '/etc/pki/tls/certs/ca-bundle.crt',     # RHEL/CentOS
            '/etc/ssl/ca-bundle.pem',               # OpenSUSE
            ssl.get_default_verify_paths().cafile,  # Python's default (may be symlink)
        ]

        ca_cert_file = None
        for cert_path in cert_paths:
            if cert_path and os.path.exists(cert_path):
                # Resolve symlinks to get the actual file path
                resolved_path = os.path.realpath(cert_path)
                if os.path.isfile(resolved_path):
                    ca_cert_file = resolved_path
                    break

        if ca_cert_file:
            # Set environment variables BEFORE creating DuckDB connection
            os.environ['CURL_CA_BUNDLE'] = ca_cert_file
            os.environ['SSL_CERT_FILE'] = ca_cert_file
            print(f"  Using SSL certificate: {ca_cert_file}")
        else:
            print("  Warning: No SSL certificate file found, connection may fail")

        # Create DuckDB connection
        con: duckdb.DuckDBPyConnection = duckdb.connect(':memory:')

        # Install and load Azure extension explicitly
        con.execute("INSTALL azure;")
        con.execute("LOAD azure;")

        # Try using curl transport adapter which should respect CURL_CA_BUNDLE env var
        try:
            con.execute("SET azure_transport_option_type='curl';")
            print("  Using CURL transport adapter for Azure")
        except Exception as e:
            print(f"  Warning: Could not set CURL transport: {e}")

        # Enable HTTP logging for debugging (optional - can be removed later)
        # con.execute("SET enable_http_logging=true;")

        # Configure Azure credentials for DuckDB
        # Use connection string with account key for authentication
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"

        con.execute(f"""
            CREATE SECRET azure_secret (
                TYPE AZURE,
                CONNECTION_STRING '{connection_string}'
            );
        """)

        return con

    def query_table(self, table_name: str, query: str) -> pl.DataFrame:
        """
        Queries a table using DuckDB and returns a Polars DataFrame.
        Reads parquet files directly from Azure Data Lake without downloading.
        For incremental tables, reads all parquet files in the directory together.
        For full tables, reads the single parquet file.

        Args:
            table_name: Name of the table (key from self.tables)
            query: SQL query to execute. Use '<table>' as a placeholder for the table reference.
                   Example: "SELECT * FROM <table> WHERE column > 100"

        Returns:
            Polars DataFrame containing the query results

        Raises:
            ValueError: If table_name not found in self.tables, or if container_name is not set in instance
            RuntimeError: If DataLake service client is not initialized or account credentials not available
            FileNotFoundError: If no parquet files are found
        """
        if not self.service_client:
            raise RuntimeError("DataLake service client is not initialized")

        if not self.container_name:
            raise ValueError("container_name must be provided during initialization")

        if table_name not in self.tables:
            raise ValueError(
                f"Table '{table_name}' not found in tables. "
                f"Available tables: {list(self.tables.keys())}"
            )

        table_info = self.tables[table_name]
        table_type = table_info['type']
        base_directory = table_info['directory']

        # Get the file system client
        file_system_client = self.service_client.get_file_system_client(self.container_name)

        try:
            # Create and configure DuckDB connection with Azure credentials
            con: duckdb.DuckDBPyConnection = self._get_duckdb_connection()

            parquet_paths = []

            if table_type == 'incremental':
                # For incremental tables, get all parquet files in the directory
                print(f"Querying incremental table '{table_name}' from {self.container_name}/{base_directory}")

                # List all files in the directory
                paths = file_system_client.get_paths(path=base_directory, recursive=False)

                for path in paths:
                    if not path.is_directory and path.name.endswith('.parquet'):
                        # Build Azure blob URL (use 'azure://' prefix like duckdb.py)
                        azure_url = f"azure://{self.container_name}/{path.name}"
                        parquet_paths.append(azure_url)

                if not parquet_paths:
                    raise FileNotFoundError(
                        f"No parquet files found in {self.container_name}/{base_directory}"
                    )

                print(f"  Found {len(parquet_paths)} parquet file(s)")

            else:
                # For full tables, read the single parquet file
                print(f"Querying full table '{table_name}' from {self.container_name}/{base_directory}")

                # Build Azure blob URL (use 'azure://' prefix like duckdb.py)
                azure_url = f"azure://{self.container_name}/{base_directory}"
                parquet_paths.append(azure_url)

            # Build the query to read from Azure
            if len(parquet_paths) == 1:
                # Single file - query it directly
                table_ref = f"read_parquet('{parquet_paths[0]}')"
            else:
                # Multiple files - use list of paths
                paths_str = "', '".join(parquet_paths)
                table_ref = f"read_parquet(['{paths_str}'])"

            # Replace '<table>' placeholder with actual table reference
            final_query = query.replace('<table>', table_ref)

            result = con.execute(final_query).pl()
            con.close()

            print(f"✓ Successfully queried table '{table_name}'")
            print(f"  Shape: {result.shape[0]:,} rows × {result.shape[1]} columns")

            return result

        except ResourceNotFoundError as e:
            raise FileNotFoundError(
                f"Path not found: {self.container_name}/{base_directory}"
            )
        except Exception as e:
            print(f"Error querying table '{table_name}': {e}")
            raise
