#!/bin/sh
set -e

envsubst '$API_PROXY_TARGET' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

exec nginx -g 'daemon off;'
