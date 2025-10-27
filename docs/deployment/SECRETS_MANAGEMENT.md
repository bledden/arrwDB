# Secrets Management Guide for arrwDB

This guide covers production-grade secrets management for arrwDB using industry-standard solutions.

## Table of Contents

- [Overview](#overview)
- [Secrets to Manage](#secrets-to-manage)
- [Option 1: AWS Secrets Manager](#option-1-aws-secrets-manager)
- [Option 2: HashiCorp Vault](#option-2-hashicorp-vault)
- [Option 3: Azure Key Vault](#option-3-azure-key-vault)
- [Option 4: Google Secret Manager](#option-4-google-secret-manager)
- [Development Setup](#development-setup)
- [Best Practices](#best-practices)
- [Rotation Strategy](#rotation-strategy)

## Overview

**Never store secrets in:**
- Source code
- Environment variable files committed to git (.env)
- Configuration files in version control
- Docker images
- Container logs

**Always store secrets in:**
- Dedicated secrets management services
- Encrypted at rest
- Access-controlled
- Audit-logged

## Secrets to Manage

arrwDB requires the following secrets:

1. **API Keys**
   - Tenant API keys
   - Service-to-service keys

2. **Embedding Service Keys**
   - Cohere API key
   - OpenAI API key (if using)
   - Custom embedding service credentials

3. **Database Credentials** (if using external DB)
   - PostgreSQL/MySQL credentials
   - Redis credentials

4. **Encryption Keys**
   - Data encryption keys
   - JWT signing keys (if implementing)

5. **Third-Party Integration Keys**
   - Monitoring service keys (Sentry, DataDog)
   - SIEM integration credentials

## Option 1: AWS Secrets Manager

### Installation

```bash
pip install boto3
```

### Store Secrets

```bash
# Create secret via AWS CLI
aws secretsmanager create-secret \
    --name arrwdb/production/cohere-api-key \
    --description "Cohere API key for arrwDB production" \
    --secret-string "your-api-key-here"

# Create secret from JSON file
aws secretsmanager create-secret \
    --name arrwdb/production/config \
    --secret-string file://secrets.json
```

### Python Integration

Create `app/utils/secrets_aws.py`:

```python
"""AWS Secrets Manager integration for arrwDB."""

import json
import boto3
from botocore.exceptions import ClientError
from functools import lru_cache
from typing import Optional, Dict, Any


class AWSSecretsManager:
    """AWS Secrets Manager client for arrwDB."""

    def __init__(self, region_name: str = "us-east-1"):
        """Initialize AWS Secrets Manager client."""
        self.client = boto3.client("secretsmanager", region_name=region_name)
        self.region = region_name

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str) -> str:
        """
        Retrieve secret from AWS Secrets Manager.

        Args:
            secret_name: Name of the secret (e.g., "arrwdb/production/api-key")

        Returns:
            Secret value as string

        Raises:
            ClientError: If secret not found or access denied
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_name)

            # Secrets can be either string or binary
            if "SecretString" in response:
                return response["SecretString"]
            else:
                import base64
                return base64.b64decode(response["SecretBinary"]).decode("utf-8")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                raise ValueError(f"Secret {secret_name} not found")
            elif error_code == "InvalidRequestException":
                raise ValueError(f"Invalid request for secret {secret_name}")
            elif error_code == "InvalidParameterException":
                raise ValueError(f"Invalid parameter for secret {secret_name}")
            elif error_code == "DecryptionFailure":
                raise RuntimeError(f"Cannot decrypt secret {secret_name}")
            elif error_code == "InternalServiceError":
                raise RuntimeError("AWS Secrets Manager internal error")
            else:
                raise

    def get_secret_json(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve secret and parse as JSON."""
        secret_value = self.get_secret(secret_name)
        return json.loads(secret_value)

    def create_secret(self, secret_name: str, secret_value: str, description: str = ""):
        """Create a new secret."""
        try:
            self.client.create_secret(
                Name=secret_name,
                Description=description,
                SecretString=secret_value,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                raise ValueError(f"Secret {secret_name} already exists")
            raise

    def update_secret(self, secret_name: str, secret_value: str):
        """Update existing secret."""
        self.client.update_secret(SecretId=secret_name, SecretString=secret_value)

    def rotate_secret(self, secret_name: str, rotation_lambda_arn: str):
        """Enable automatic rotation for secret."""
        self.client.rotate_secret(
            SecretId=secret_name,
            RotationLambdaARN=rotation_lambda_arn,
            RotationRules={"AutomaticallyAfterDays": 30},
        )

    def delete_secret(self, secret_name: str, recovery_window_days: int = 30):
        """
        Delete secret with recovery window.

        Args:
            secret_name: Name of secret to delete
            recovery_window_days: Days before permanent deletion (7-30)
        """
        self.client.delete_secret(
            SecretId=secret_name,
            RecoveryWindowInDays=recovery_window_days,
        )


# Global instance
_secrets_manager: Optional[AWSSecretsManager] = None


def get_secrets_manager() -> AWSSecretsManager:
    """Get global AWS Secrets Manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        from app.config import settings
        _secrets_manager = AWSSecretsManager(region_name=settings.AWS_REGION)
    return _secrets_manager


def get_secret(secret_name: str) -> str:
    """Convenience function to get a secret."""
    return get_secrets_manager().get_secret(secret_name)
```

### Usage in arrwDB

Update `app/config.py`:

```python
from app.utils.secrets_aws import get_secret

class Settings(BaseSettings):
    # ...existing settings...

    # Secrets management
    USE_SECRETS_MANAGER: bool = False
    AWS_REGION: str = "us-east-1"
    SECRETS_PREFIX: str = "arrwdb/production"

    @property
    def cohere_api_key(self) -> str:
        """Get Cohere API key from secrets manager or environment."""
        if self.USE_SECRETS_MANAGER:
            return get_secret(f"{self.SECRETS_PREFIX}/cohere-api-key")
        return os.getenv("COHERE_API_KEY", "")
```

### IAM Policy

Grant arrwDB EC2 instance/ECS task this IAM policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:arrwdb/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": [
        "arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"
      ]
    }
  ]
}
```

## Option 2: HashiCorp Vault

### Installation

```bash
# Install Vault CLI
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Install Python client
pip install hvac
```

### Store Secrets

```bash
# Start Vault dev server (development only!)
vault server -dev

# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'

# Login (use root token from server output)
vault login

# Store secrets
vault kv put secret/arrwdb/production/cohere api_key="your-key-here"

# Store JSON secrets
vault kv put secret/arrwdb/production/database \
    username="admin" \
    password="secure-password" \
    host="db.example.com" \
    port="5432"
```

### Python Integration

Create `app/utils/secrets_vault.py`:

```python
"""HashiCorp Vault integration for arrwDB."""

import hvac
from functools import lru_cache
from typing import Dict, Any, Optional


class VaultSecretsManager:
    """HashiCorp Vault client for arrwDB."""

    def __init__(
        self,
        vault_url: str = "http://127.0.0.1:8200",
        token: Optional[str] = None,
        mount_point: str = "secret",
    ):
        """
        Initialize Vault client.

        Args:
            vault_url: Vault server URL
            token: Vault token (or use AppRole/Kubernetes auth)
            mount_point: KV secrets engine mount point
        """
        self.client = hvac.Client(url=vault_url, token=token)
        self.mount_point = mount_point

        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")

    @lru_cache(maxsize=128)
    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Retrieve secret from Vault.

        Args:
            path: Secret path (e.g., "arrwdb/production/cohere")
            key: Optional key within secret (e.g., "api_key")

        Returns:
            Secret value or dict of all values if key not specified
        """
        try:
            # KV v2 (default)
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            data = response["data"]["data"]

            if key:
                return data[key]
            return data

        except hvac.exceptions.InvalidPath:
            raise ValueError(f"Secret path {path} not found")
        except Exception as e:
            raise RuntimeError(f"Error retrieving secret: {e}")

    def create_secret(self, path: str, secret_data: Dict[str, Any]):
        """Create or update secret."""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path, secret=secret_data, mount_point=self.mount_point
        )

    def delete_secret(self, path: str):
        """Delete secret (soft delete in KV v2)."""
        self.client.secrets.kv.v2.delete_latest_version_of_secret(
            path=path, mount_point=self.mount_point
        )

    def list_secrets(self, path: str = "") -> list:
        """List secrets at path."""
        response = self.client.secrets.kv.v2.list_secrets(
            path=path, mount_point=self.mount_point
        )
        return response["data"]["keys"]

    @classmethod
    def from_approle(cls, vault_url: str, role_id: str, secret_id: str, **kwargs):
        """
        Authenticate using AppRole.

        Args:
            vault_url: Vault server URL
            role_id: AppRole role ID
            secret_id: AppRole secret ID
        """
        client = hvac.Client(url=vault_url)
        client.auth.approle.login(role_id=role_id, secret_id=secret_id)
        return cls(vault_url=vault_url, token=client.token, **kwargs)


# Global instance
_vault_manager: Optional[VaultSecretsManager] = None


def get_vault_manager() -> VaultSecretsManager:
    """Get global Vault manager instance."""
    global _vault_manager
    if _vault_manager is None:
        from app.config import settings
        _vault_manager = VaultSecretsManager(
            vault_url=settings.VAULT_URL, token=settings.VAULT_TOKEN
        )
    return _vault_manager
```

### AppRole Authentication (Recommended)

```bash
# Enable AppRole auth
vault auth enable approle

# Create policy for arrwDB
vault policy write arrwdb-policy - <<EOF
path "secret/data/arrwdb/*" {
  capabilities = ["read", "list"]
}
EOF

# Create AppRole
vault write auth/approle/role/arrwdb \
    secret_id_ttl=24h \
    token_ttl=1h \
    token_max_ttl=24h \
    policies="arrwdb-policy"

# Get Role ID
vault read auth/approle/role/arrwdb/role-id

# Generate Secret ID
vault write -f auth/approle/role/arrwdb/secret-id
```

## Option 3: Azure Key Vault

### Installation

```bash
pip install azure-identity azure-keyvault-secrets
```

### Store Secrets

```bash
# Create Key Vault
az keyvault create \
    --name arrwdb-vault \
    --resource-group my-resource-group \
    --location eastus

# Store secret
az keyvault secret set \
    --vault-name arrwdb-vault \
    --name cohere-api-key \
    --value "your-key-here"
```

### Python Integration

Create `app/utils/secrets_azure.py`:

```python
"""Azure Key Vault integration for arrwDB."""

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from functools import lru_cache
from typing import Optional


class AzureSecretsManager:
    """Azure Key Vault client for arrwDB."""

    def __init__(self, vault_url: str):
        """
        Initialize Azure Key Vault client.

        Args:
            vault_url: Key Vault URL (e.g., "https://arrwdb-vault.vault.azure.net")
        """
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Azure Key Vault."""
        secret = self.client.get_secret(secret_name)
        return secret.value

    def set_secret(self, secret_name: str, secret_value: str):
        """Create or update secret."""
        self.client.set_secret(secret_name, secret_value)

    def delete_secret(self, secret_name: str):
        """Delete secret."""
        self.client.begin_delete_secret(secret_name).wait()

    def list_secrets(self):
        """List all secret names."""
        return [secret.name for secret in self.client.list_properties_of_secrets()]
```

### Managed Identity Setup

Grant your Azure VM/Container Instance access:

```bash
# Assign managed identity
az vm identity assign --name myVM --resource-group myRG

# Grant Key Vault permissions
az keyvault set-policy \
    --name arrwdb-vault \
    --object-id <managed-identity-object-id> \
    --secret-permissions get list
```

## Option 4: Google Secret Manager

### Installation

```bash
pip install google-cloud-secret-manager
```

### Store Secrets

```bash
# Create secret
gcloud secrets create cohere-api-key \
    --replication-policy="automatic"

# Add secret version
echo -n "your-key-here" | gcloud secrets versions add cohere-api-key --data-file=-
```

### Python Integration

Create `app/utils/secrets_gcp.py`:

```python
"""Google Secret Manager integration for arrwDB."""

from google.cloud import secretmanager
from functools import lru_cache
from typing import Optional


class GCPSecretsManager:
    """Google Secret Manager client for arrwDB."""

    def __init__(self, project_id: str):
        """Initialize GCP Secret Manager client."""
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, version: str = "latest") -> str:
        """
        Retrieve secret from Google Secret Manager.

        Args:
            secret_name: Name of the secret
            version: Version to retrieve (default: "latest")
        """
        name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

    def create_secret(self, secret_name: str, secret_value: str):
        """Create a new secret."""
        parent = f"projects/{self.project_id}"

        # Create secret
        secret = self.client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_name,
                "secret": {"replication": {"automatic": {}}},
            }
        )

        # Add secret version
        self.client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )

    def delete_secret(self, secret_name: str):
        """Delete secret."""
        name = f"projects/{self.project_id}/secrets/{secret_name}"
        self.client.delete_secret(request={"name": name})
```

### Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create arrwdb-sa

# Grant permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:arrwdb-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Development Setup

For local development, use `.env` files (NOT committed to git):

Create `.env.local`:

```bash
# arrwDB Configuration
DEBUG=true

# Embedding Service (for local dev only!)
COHERE_API_KEY=dev-key-here

# Secrets Manager (disabled in dev)
USE_SECRETS_MANAGER=false
```

Add to `.gitignore`:

```
.env
.env.local
.env.*.local
secrets/
*.pem
*.key
```

## Best Practices

### 1. Principle of Least Privilege

Grant only necessary permissions:

```
✅ arrwDB can READ secrets
❌ arrwDB cannot CREATE/DELETE secrets
```

### 2. Secrets Rotation

Rotate secrets regularly:

- **API Keys**: Every 90 days
- **Database Passwords**: Every 30 days
- **Encryption Keys**: Annually (with key versioning)

### 3. Audit Logging

Enable audit logs for all secret access:

```python
from app.auth.audit import get_audit_logger

def get_secret_with_audit(secret_name: str) -> str:
    """Get secret and log access."""
    audit_logger = get_audit_logger()

    try:
        secret = get_secret(secret_name)
        audit_logger.log(AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.ADMIN_ACTION,
            severity=Severity.INFO,
            message=f"Secret accessed: {secret_name}",
        ))
        return secret
    except Exception as e:
        audit_logger.log(AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.ADMIN_ACTION,
            severity=Severity.ERROR,
            message=f"Secret access failed: {secret_name}",
        ))
        raise
```

### 4. Secret Versioning

Always use versioned secrets for rollback capability:

```python
# AWS
get_secret(f"{prefix}/api-key", version_id="v2")

# Vault
vault kv get -version=2 secret/arrwdb/api-key

# GCP
get_secret("api-key", version="2")
```

### 5. Environment Separation

Use different secrets for each environment:

```
- arrwdb/development/*
- arrwdb/staging/*
- arrwdb/production/*
```

### 6. Secret Caching

Cache secrets with TTL to reduce API calls:

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedSecretsManager:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache = {}

    def get_secret(self, name: str) -> str:
        if name in self.cache:
            value, timestamp = self.cache[name]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl):
                return value

        # Fetch from secrets manager
        value = fetch_secret(name)
        self.cache[name] = (value, datetime.utcnow())
        return value
```

## Rotation Strategy

### Manual Rotation

```bash
# 1. Generate new secret
NEW_KEY=$(openssl rand -base64 32)

# 2. Store new secret version
aws secretsmanager put-secret-value \
    --secret-id arrwdb/production/api-key \
    --secret-string "$NEW_KEY"

# 3. Deploy arrwDB with new secret
# 4. Verify functionality
# 5. Revoke old secret
```

### Automatic Rotation

AWS Secrets Manager supports automatic rotation:

```python
import boto3

def lambda_handler(event, context):
    """Lambda function for secret rotation."""
    service_client = boto3.client("secretsmanager")
    arn = event["SecretId"]
    token = event["ClientRequestToken"]
    step = event["Step"]

    if step == "createSecret":
        # Generate new secret
        new_secret = generate_new_api_key()
        service_client.put_secret_value(
            SecretId=arn,
            ClientRequestToken=token,
            SecretString=new_secret,
            VersionStages=["AWSPENDING"],
        )

    elif step == "setSecret":
        # Update service with new secret
        pass

    elif step == "testSecret":
        # Test new secret
        pass

    elif step == "finishSecret":
        # Finalize rotation
        service_client.update_secret_version_stage(
            SecretId=arn,
            VersionStage="AWSCURRENT",
            MoveToVersionId=token,
            RemoveFromVersionId=current_version,
        )
```

## Security Checklist

- [ ] Secrets never in source code
- [ ] `.env` files in `.gitignore`
- [ ] Secrets manager enabled for production
- [ ] Least privilege IAM/RBAC policies
- [ ] Audit logging enabled
- [ ] Secrets rotation schedule defined
- [ ] Environment separation (dev/staging/prod)
- [ ] Secret versioning enabled
- [ ] Cache TTL configured appropriately
- [ ] Backup/recovery procedure documented
- [ ] Team trained on secrets management

## Additional Resources

- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [Azure Key Vault Documentation](https://docs.microsoft.com/en-us/azure/key-vault/)
- [Google Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
