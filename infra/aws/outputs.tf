# The values worth reading after an apply, and the ones to paste back into the
# variable file so the next apply is reproducible.

output "instance_id" {
  description = "Instance id of the experiment host."
  value       = aws_instance.host.id
}

output "public_ip" {
  description = "Public IPv4 address of the experiment host."
  value       = aws_instance.host.public_ip
}

output "ssh_command" {
  description = "Ready to paste once the instance has finished booting."
  value       = "ssh ${var.ssh_user}@${aws_instance.host.public_ip}"
}

output "spot_instance_request_id" {
  description = <<-EOT
    The Spot request behind the instance. Useful when the request is not
    fulfilled and nothing appears, because the request status says why.
  EOT
  value       = aws_instance.host.spot_instance_request_id
}

output "resolved_ami_id" {
  description = <<-EOT
    The AMI actually used. Copy this into ami_id in terraform.tfvars so later
    applies pin the same image instead of picking up a newer one.
  EOT
  value       = local.ami_id
}

output "instance_type" {
  description = "Instance type actually requested."
  value       = aws_instance.host.instance_type
}

output "availability_zone" {
  description = "Zone the host landed in. Spot capacity varies by zone."
  value       = data.aws_subnet.chosen.availability_zone
}

output "subnet_id" {
  description = "Subnet used. Set this in terraform.tfvars to pin a zone next time."
  value       = local.subnet_id
}

output "security_group_id" {
  description = "Security group allowing SSH from the supplied CIDR."
  value       = aws_security_group.host.id
}

output "root_volume_size_gb" {
  description = "Root EBS size requested."
  value       = var.root_volume_size_gb
}

output "key_pair_name" {
  description = "Registered key pair name."
  value       = aws_key_pair.host.key_name
}
