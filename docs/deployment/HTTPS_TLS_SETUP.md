# HTTPS/TLS Deployment Guide for arrwDB

This guide covers production-grade HTTPS/TLS configuration for arrwDB using industry-standard reverse proxies.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Option 1: nginx + Let's Encrypt](#option-1-nginx--lets-encrypt)
- [Option 2: Caddy (Auto HTTPS)](#option-2-caddy-auto-https)
- [Option 3: Traefik](#option-3-traefik)
- [SSL/TLS Best Practices](#ssltls-best-practices)
- [Certificate Management](#certificate-management)
- [Testing & Verification](#testing--verification)
- [Troubleshooting](#troubleshooting)

## Overview

arrwDB runs as a Python/FastAPI application on port 8000 by default. For production deployments, you should:

1. **Never expose port 8000 directly to the internet**
2. **Always use a reverse proxy** (nginx, Caddy, Traefik)
3. **Enable HTTPS with valid TLS certificates**
4. **Configure HSTS headers** (already handled by arrwDB security middleware)
5. **Implement rate limiting** at the proxy level

## Prerequisites

- Domain name pointing to your server (e.g., `api.your domain.com`)
- Server with public IP address
- Ports 80 (HTTP) and 443 (HTTPS) open in firewall
- arrwDB running on `localhost:8000`

## Option 1: nginx + Let's Encrypt

### Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx

# CentOS/RHEL
sudo yum install nginx certbot python3-certbot-nginx
```

### nginx Configuration

Create `/etc/nginx/sites-available/arrwdb`:

```nginx
# Upstream definition for arrwDB
upstream arrwdb_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name api.yourdomain.com;

    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.yourdomain.com;

    # SSL certificates (configured by certbot)
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # SSL configuration (Mozilla Intermediate)
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/api.yourdomain.com/chain.pem;

    # Security headers (arrwDB also sets these, but defense in depth)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Request size limit (adjust based on your needs)
    client_max_body_size 100M;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req zone=api_limit burst=200 nodelay;

    # Proxy settings
    location / {
        proxy_pass http://arrwdb_backend;
        proxy_http_version 1.1;

        # Preserve original request information
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;
    }

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://arrwdb_backend;
        access_log off;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        # Only allow from internal IPs
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://arrwdb_backend;
    }
}
```

### Enable Configuration

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/arrwdb /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### Obtain SSL Certificate

```bash
# Get certificate (this will auto-configure nginx)
sudo certbot --nginx -d api.yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

### Auto-Renewal Setup

Certbot automatically installs a systemd timer for renewal:

```bash
# Check renewal timer status
sudo systemctl status certbot.timer

# Manual renewal (if needed)
sudo certbot renew
```

## Option 2: Caddy (Auto HTTPS)

Caddy automatically obtains and renews certificates from Let's Encrypt.

### Installation

```bash
# Ubuntu/Debian
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

### Caddyfile Configuration

Create `/etc/caddy/Caddyfile`:

```caddy
# Global options
{
    email admin@yourdomain.com
}

# arrwDB API
api.yourdomain.com {
    # Automatic HTTPS (no configuration needed!)

    # Rate limiting
    rate_limit {
        zone api_zone {
            key {remote_host}
            events 100
            window 1s
        }
    }

    # Request size limit
    request_body {
        max_size 100MB
    }

    # Reverse proxy to arrwDB
    reverse_proxy localhost:8000 {
        # Health check
        health_uri /health
        health_interval 10s
        health_timeout 5s

        # Headers
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }

    # Security headers (arrwDB also sets these)
    header {
        Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"
        X-Frame-Options "DENY"
        X-Content-Type-Options "nosniff"
        X-XSS-Protection "1; mode=block"
    }

    # Metrics endpoint restrictions
    @metrics {
        path /metrics
        not remote_ip 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16
    }
    respond @metrics 403

    # Logging
    log {
        output file /var/log/caddy/arrwdb-access.log
        format json
    }
}
```

### Start Caddy

```bash
# Start Caddy
sudo systemctl start caddy

# Enable on boot
sudo systemctl enable caddy

# Check status
sudo systemctl status caddy

# View logs
sudo journalctl -u caddy -f
```

Caddy will automatically:
- Obtain Let's Encrypt certificates
- Renew certificates before expiration
- Redirect HTTP to HTTPS
- Enable HTTP/2

## Option 3: Traefik

Traefik is a modern cloud-native proxy with automatic HTTPS via Let's Encrypt.

### Docker Compose Setup

Create `docker-compose.traefik.yml`:

```yaml
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    command:
      # API and dashboard
      - "--api.dashboard=true"
      # Entry points
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      # Let's Encrypt
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.letsencrypt.acme.email=admin@yourdomain.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
      # Docker provider
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./letsencrypt:/letsencrypt
    labels:
      - "traefik.enable=true"
      # Dashboard (secure this!)
      - "traefik.http.routers.dashboard.rule=Host(`traefik.yourdomain.com`)"
      - "traefik.http.routers.dashboard.entrypoints=websecure"
      - "traefik.http.routers.dashboard.tls.certresolver=letsencrypt"
      - "traefik.http.routers.dashboard.service=api@internal"

  arrwdb:
    image: arrwdb:latest
    labels:
      - "traefik.enable=true"
      # HTTP to HTTPS redirect
      - "traefik.http.routers.arrwdb-http.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.arrwdb-http.entrypoints=web"
      - "traefik.http.routers.arrwdb-http.middlewares=redirect-to-https"
      # HTTPS router
      - "traefik.http.routers.arrwdb.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.arrwdb.entrypoints=websecure"
      - "traefik.http.routers.arrwdb.tls.certresolver=letsencrypt"
      - "traefik.http.routers.arrwdb.service=arrwdb"
      # Service
      - "traefik.http.services.arrwdb.loadbalancer.server.port=8000"
      # Middleware
      - "traefik.http.middlewares.redirect-to-https.redirectscheme.scheme=https"
      - "traefik.http.middlewares.redirect-to-https.redirectscheme.permanent=true"
```

## SSL/TLS Best Practices

### 1. Use Modern TLS Versions

- **Enable**: TLS 1.2 and TLS 1.3
- **Disable**: TLS 1.0, TLS 1.1, SSLv2, SSLv3

### 2. Strong Cipher Suites

Recommend Mozilla's "Intermediate" configuration for broad compatibility:

```
ECDHE-ECDSA-AES128-GCM-SHA256
ECDHE-RSA-AES128-GCM-SHA256
ECDHE-ECDSA-AES256-GCM-SHA384
ECDHE-RSA-AES256-GCM-SHA384
ECDHE-ECDSA-CHACHA20-POLY1305
ECDHE-RSA-CHACHA20-POLY1305
```

### 3. HSTS Configuration

Always enable HTTP Strict Transport Security:

```
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
```

Consider adding your domain to the [HSTS Preload List](https://hstspreload.org/).

### 4. OCSP Stapling

Enable OCSP stapling to improve SSL/TLS handshake performance and privacy.

### 5. Certificate Pinning (Optional)

For high-security environments, implement certificate pinning:

```
Public-Key-Pins: pin-sha256="base64=="; pin-sha256="base64=="; max-age=5184000
```

## Certificate Management

### Automatic Renewal

All three options (nginx + certbot, Caddy, Traefik) automatically renew certificates.

### Manual Renewal (nginx + certbot)

```bash
# Renew all certificates
sudo certbot renew

# Renew specific certificate
sudo certbot renew --cert-name api.yourdomain.com

# Force renewal (for testing)
sudo certbot renew --force-renewal
```

### Monitoring Certificate Expiration

Create a monitoring script:

```bash
#!/bin/bash
# check-cert-expiry.sh

DOMAIN="api.yourdomain.com"
DAYS_WARN=30

EXPIRY=$(echo | openssl s_client -servername $DOMAIN -connect $DOMAIN:443 2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2)
EXPIRY_EPOCH=$(date -d "$EXPIRY" +%s)
NOW_EPOCH=$(date +%s)
DAYS_LEFT=$(( ($EXPIRY_EPOCH - $NOW_EPOCH) / 86400 ))

if [ $DAYS_LEFT -lt $DAYS_WARN ]; then
    echo "WARNING: Certificate expires in $DAYS_LEFT days!"
    # Send alert (email, Slack, PagerDuty, etc.)
fi
```

### Backup Certificates

```bash
# Backup Let's Encrypt certificates
sudo tar -czf letsencrypt-backup-$(date +%Y%m%d).tar.gz /etc/letsencrypt/

# Restore from backup
sudo tar -xzf letsencrypt-backup-YYYYMMDD.tar.gz -C /
```

## Testing & Verification

### 1. SSL Labs Test

Test your configuration at [SSL Labs](https://www.ssllabs.com/ssltest/):

```
https://www.ssllabs.com/ssltest/analyze.html?d=api.yourdomain.com
```

Target grade: **A or A+**

### 2. Command Line Test

```bash
# Test TLS connection
openssl s_client -connect api.yourdomain.com:443 -servername api.yourdomain.com

# Check certificate validity
echo | openssl s_client -servername api.yourdomain.com -connect api.yourdomain.com:443 2>/dev/null | openssl x509 -noout -dates

# Test HTTP/2 support
curl -I --http2 https://api.yourdomain.com/health

# Verify HSTS header
curl -I https://api.yourdomain.com | grep -i strict-transport
```

### 3. Automated Testing

Use [testssl.sh](https://testssl.sh/) for comprehensive testing:

```bash
# Install testssl.sh
git clone --depth 1 https://github.com/drwetter/testssl.sh.git
cd testssl.sh

# Run full test
./testssl.sh --full https://api.yourdomain.com
```

## Troubleshooting

### Certificate Not Trusted

**Problem**: Browser shows "Certificate not trusted" error.

**Solutions**:
- Ensure Let's Encrypt chain is properly configured
- Check that your domain DNS is correct
- Verify firewall allows port 443
- Try clearing browser cache

### Certificate Renewal Fails

**Problem**: Certbot fails to renew certificates.

**Solutions**:
```bash
# Check certbot logs
sudo journalctl -u certbot -n 50

# Verify HTTP challenge path is accessible
curl http://api.yourdomain.com/.well-known/acme-challenge/test

# Test renewal manually
sudo certbot renew --dry-run --debug
```

### Mixed Content Warnings

**Problem**: HTTPS page loads HTTP resources.

**Solutions**:
- Ensure arrwDB uses HTTPS URLs in responses
- Check `X-Forwarded-Proto` header is set correctly
- Update any hardcoded HTTP URLs to HTTPS

### Performance Issues

**Problem**: Slow HTTPS connections.

**Solutions**:
- Enable HTTP/2
- Enable OCSP stapling
- Increase SSL session cache size
- Use session resumption
- Enable TLS 1.3

### WebSocket Connection Fails

**Problem**: WebSocket connections fail over HTTPS.

**Solutions**:
```nginx
# Ensure nginx has WebSocket support
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_http_version 1.1;
```

## Security Checklist

- [ ] TLS 1.2+ only (disable TLS 1.0/1.1)
- [ ] Strong cipher suites configured
- [ ] HSTS header enabled with long max-age
- [ ] OCSP stapling enabled
- [ ] Certificate auto-renewal configured
- [ ] Certificate expiration monitoring set up
- [ ] Tested with SSL Labs (A/A+ grade)
- [ ] HTTP to HTTPS redirect configured
- [ ] Rate limiting enabled at proxy level
- [ ] Request size limits configured
- [ ] Security headers configured
- [ ] Firewall rules configured (allow 80, 443 only)
- [ ] arrwDB not directly exposed to internet
- [ ] Regular security updates scheduled

## Additional Resources

- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [OWASP TLS Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [nginx SSL/TLS Guide](https://nginx.org/en/docs/http/configuring_https_servers.html)
- [Caddy Documentation](https://caddyserver.com/docs/)
- [Traefik Documentation](https://doc.traefik.io/traefik/)
