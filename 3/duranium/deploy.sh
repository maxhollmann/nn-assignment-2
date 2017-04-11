#!/usr/bin/env bash

rsync -r . duranium:3 --exclude __pycache__ --exclude out --exclude "#*" --exclude ".#*"
