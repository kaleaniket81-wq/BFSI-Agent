# infra/terraform/main.tf
# Provisions cloud resources for production deployment.
# For local dev, everything runs via docker-compose.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ── AWS provider ──────────────────────────────────────────────────────────────
provider "aws" {
  region = var.aws_region   # ap-south-1 (Mumbai) — closest to Pune
}

# ── RDS PostgreSQL with pgvector ──────────────────────────────────────────────
# Stores document chunks + embeddings
resource "aws_db_instance" "bfsi_postgres" {
  count               = var.deploy_rds ? 1 : 0
  identifier          = "${var.project_name}-db"
  engine              = "postgres"
  engine_version      = "16.2"
  instance_class      = "db.t3.micro"   # free tier eligible
  allocated_storage   = 20
  db_name             = "bfsi_intelligence"
  username            = var.db_username
  password            = var.db_password
  skip_final_snapshot = true
  publicly_accessible = false
  tags                = { Project = var.project_name, Environment = "production" }
}

# ── S3 bucket — document storage ──────────────────────────────────────────────
# Raw PDF/DOCX files before ingestion
resource "aws_s3_bucket" "documents" {
  bucket = "${var.project_name}-documents-${random_id.suffix.hex}"
  tags   = { Project = var.project_name }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── ElastiCache Redis — embedding + query cache ───────────────────────────────
resource "aws_elasticache_cluster" "redis" {
  count                = var.deploy_redis ? 1 : 0
  cluster_id           = "${var.project_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  tags                 = { Project = var.project_name }
}

# ── ECR — Docker image registry ───────────────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name = "${var.project_name}-api"
  tags = { Project = var.project_name }
}

resource "aws_ecr_repository" "worker" {
  name = "${var.project_name}-worker"
  tags = { Project = var.project_name }
}

# ── IAM role for Bedrock access ───────────────────────────────────────────────
# Attach this role to EC2/ECS — no API keys needed in env vars
resource "aws_iam_role" "bfsi_app_role" {
  name = "${var.project_name}-app-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_access" {
  name = "bedrock-access"
  role = aws_iam_role.bfsi_app_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:ListFoundationModels"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.documents.arn}/*"
      }
    ]
  })
}

resource "random_id" "suffix" { byte_length = 4 }
