variable "project_name"   { default = "bfsi-doc-intelligence" }
variable "aws_region"     { default = "ap-south-1" }   # Mumbai
variable "db_username"    { default = "bfsi_user" }
variable "db_password"    { sensitive = true }
variable "deploy_rds"     { default = false }           # set true to provision RDS
variable "deploy_redis"   { default = false }           # set true to provision ElastiCache
