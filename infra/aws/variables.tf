# Inputs. The AWS choices live here and are supplied, not invented: region,
# purchasing model, instance type and AMI family are all externally decided and
# only repeated here as defaults so the example file is smaller.

variable "region" {
  description = "AWS region for the experiment host."
  type        = string
  default     = "ap-south-1"
}

variable "name_prefix" {
  description = "Prefix for the names of every resource this configuration creates."
  type        = string
  default     = "voice-inference-lab"
}

variable "instance_type" {
  description = "Instance type for the experiment host. 1x A10G, 8 vCPU, 32 GiB."
  type        = string
  default     = "g5.2xlarge"
}

variable "spot_max_price" {
  description = <<-EOT
    Leave MaxPrice unset by default. AWS recommends not specifying a maximum
    price for Spot Instances. Set a value only if you deliberately want to cap
    the hourly price.
  EOT
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# AMI
# ---------------------------------------------------------------------------

variable "ami_id" {
  description = <<-EOT
    Pinned AMI id. This is the reproducible path and should be set once a first
    apply has reported one. Leave empty to look up the newest image matching
    ami_name_filter, which will drift as the family gains images.
  EOT
  type        = string
  default     = ""
}

variable "ami_name_filter" {
  description = "Name pattern used only when ami_id is empty."
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
}

variable "ami_owners" {
  description = "AMI owners to search when ami_id is empty."
  type        = list(string)
  default     = ["amazon"]
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

variable "subnet_id" {
  description = <<-EOT
    Subnet to launch into. Leave empty to take one subnet from the default VPC.
    Set this to pin an availability zone, which can matter when Spot capacity is
    scarce in one zone.
  EOT
  type        = string
  default     = ""
}

variable "ssh_ingress_cidr" {
  description = <<-EOT
    Required. The only CIDR allowed to reach port 22. Supply your own address,
    normally as a /32.
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.ssh_ingress_cidr))
    error_message = "ssh_ingress_cidr must be a valid IPv4 CIDR block with a prefix length, such as /32 for a single address."
  }

  validation {
    condition     = !can(regex("^203\\.0\\.113\\.", var.ssh_ingress_cidr))
    error_message = "ssh_ingress_cidr is inside the documentation range 203.0.113.0/24, so it is almost certainly not your address. Supply your real address."
  }
}

# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------

variable "ssh_public_key_path" {
  description = "Local path to the public key to authorise on the host."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_user" {
  description = "Login user for the chosen AMI. Used only to render the SSH output."
  type        = string
  default     = "ubuntu"
}

# ---------------------------------------------------------------------------
# Root volume
# ---------------------------------------------------------------------------

variable "root_volume_size_gb" {
  description = <<-EOT
    Root EBS size in GiB. This must be at least the size of the AMI's root
    snapshot, or the apply is rejected. The experiment data belongs on the
    instance's local disk, not here.
  EOT
  type        = number
  default     = 100
}

variable "root_volume_type" {
  description = "Root EBS volume type."
  type        = string
  default     = "gp3"
}

variable "root_volume_encrypted" {
  description = "Encrypt the root EBS volume at rest."
  type        = bool
  default     = true
}

# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

variable "tags" {
  description = "Tags applied to every resource that supports them."
  type        = map(string)
  default = {
    project = "voice-inference-lab"
    purpose = "gpu-experiment-host"
  }
}
