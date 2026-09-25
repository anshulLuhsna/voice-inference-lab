# A single GPU experiment host.
#
# Terraform reproduces the machine. It does not reproduce the workload: the
# runtime install and the experiments live elsewhere in this repository and are
# run over SSH. Nothing here downloads a model, configures Moshi, or deploys
# application code.
#
#   terraform apply      -> the machine exists
#   ssh / install runtime -> see aws/README.md
#   run the experiment
#   copy results off
#   terraform destroy    -> the machine is gone
#
# This file makes no AWS choices of its own beyond what the variables say. The
# region, the instance type, the AMI family and the purchasing model are all
# supplied.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # Validated against 6.66.0. The lock file is committed and pins that exact
      # version with its hashes, so the provider is reproducible too.
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region

  # No credentials, no account id, no secrets in this file or in the example
  # variable file. Use the standard AWS environment variables or a named
  # profile via AWS_PROFILE.
}

# ---------------------------------------------------------------------------
# Networking. The default VPC and one of its subnets are reused. No VPC, no
# route table, no internet gateway, and no NAT are created, because the default
# networking already provides everything a single public host needs.
#
# If this account has no default VPC, set var.subnet_id instead.
# ---------------------------------------------------------------------------
data "aws_vpc" "default" {
  count   = var.subnet_id == "" ? 1 : 0
  default = true
}

data "aws_subnets" "default" {
  count = var.subnet_id == "" ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default[0].id]
  }
}

locals {
  subnet_id = var.subnet_id != "" ? var.subnet_id : data.aws_subnets.default[0].ids[0]
}

# Read back the chosen subnet so the security group lands in the right VPC and
# the availability zone can be reported. Spot capacity varies by zone, so the
# zone is worth seeing before the first apply.
data "aws_subnet" "chosen" {
  id = local.subnet_id
}

# ---------------------------------------------------------------------------
# AMI selection.
#
# Supply ami_id to pin an image exactly. That is the reproducible path.
#
# Left empty, the newest image matching the supplied family name is looked up
# instead. That is convenient but not reproducible over time, because the family
# gains new images, so the resolved id is reported in the outputs and should be
# pasted back into the variable file afterwards.
# ---------------------------------------------------------------------------
data "aws_ami" "selected" {
  count = var.ami_id == "" ? 1 : 0

  most_recent = true
  owners      = var.ami_owners

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.selected[0].id
}

# ---------------------------------------------------------------------------
# SSH key. The public key is read from a local file and registered as a key
# pair. No private key is handled, stored, or transmitted anywhere.
# ---------------------------------------------------------------------------
resource "aws_key_pair" "host" {
  key_name   = "${var.name_prefix}-key"
  public_key = file(pathexpand(var.ssh_public_key_path))

  tags = merge(var.tags, { Name = "${var.name_prefix}-key" })
}

# ---------------------------------------------------------------------------
# Security group. Exactly one ingress rule, from the CIDR you supply, on port
# 22. There is no 0.0.0.0/0 ingress rule anywhere in this configuration and no
# default that would produce one.
#
# Egress is unrestricted because the first thing the machine does is install a
# pinned runtime from the network.
# ---------------------------------------------------------------------------
resource "aws_security_group" "host" {
  name_prefix = "${var.name_prefix}-ssh-"
  description = "SSH to the ${var.name_prefix} experiment host"
  vpc_id      = data.aws_subnet.chosen.vpc_id

  ingress {
    description = "SSH from the supplied CIDR"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    description = "Runtime install and model download"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-ssh" })

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------
# The machine.
#
# Spot is represented as a market option on an ordinary instance rather than as
# a separate request resource, so that `terraform destroy` has the same clean
# terminate path as any other instance. The purchase model is one-time and the
# interruption behaviour is terminate, which matches a machine that is meant to
# be disposable.
# ---------------------------------------------------------------------------
resource "aws_instance" "host" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.host.id]
  key_name                    = aws_key_pair.host.key_name
  associate_public_ip_address = true

  # Instance metadata requires IMDSv2. The host is reachable over SSH from your
  # CIDR, so a metadata service that answers unauthenticated requests would be a
  # genuine risk. Remove this block if it breaks a tool you rely on.
  metadata_options {
    http_tokens = "required"
  }

  # Root volume. It is deleted on termination, so nothing is left behind by
  # `terraform destroy`. The attached instance store is ephemeral by nature.
  root_block_device {
    volume_type           = var.root_volume_type
    volume_size           = var.root_volume_size_gb
    encrypted             = var.root_volume_encrypted
    delete_on_termination = true
  }

  instance_market_options {
    market_type = "spot"

    spot_options {
      spot_instance_type = "one-time"
      # Valid with a one-time request, and required for a machine that is meant
      # to vanish rather than come back.
      instance_interruption_behavior = "terminate"
      # Leave MaxPrice unset by default. AWS recommends not specifying a
      # maximum price for Spot Instances.
      max_price = var.spot_max_price != "" ? var.spot_max_price : null
    }
  }

  # A Spot request can queue when capacity is scarce, and the default create
  # timeout of 10 minutes cancels a request that is still waiting rather than
  # one that was rejected. ap-south-1b queues: it sat for over five minutes
  # without resolving either way, while ap-south-1a rejected the request
  # outright. This lets a queued request survive long enough to be fulfilled.
  timeouts {
    create = "30m"
  }

  tags = merge(var.tags, { Name = var.name_prefix })
}
