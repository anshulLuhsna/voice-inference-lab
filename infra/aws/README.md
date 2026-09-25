# AWS GPU experiment host — frozen

This path reproduces a single GPU experiment host with Terraform. It is
complete enough to create and destroy a machine, and it is **frozen unfinished**
by decision, not abandoned by accident.

## Outcome

The host was never obtained. No g5.2xlarge Spot capacity was available in
ap-south-1 in any zone that offers the instance type.

| Zone | Instance type offered | Observed |
| --- | --- | --- |
| ap-south-1a | yes | rejected immediately, `InsufficientInstanceCapacity` |
| ap-south-1b | yes | SDK retried ~47 minutes, then `InsufficientInstanceCapacity` |
| ap-south-1c | **no** | `Unsupported`, the type is not offered in this zone |

The Spot placement score API was also consulted and was uninformative: it
returned 1 for both 1a and 1b, and nothing at all for 1c, which turned out to
mean "the type is not offered here" rather than "capacity is scarce". A score of
1 is not a prediction of success, since 1a scored 1 and failed outright.

**Decision:** AWS is frozen and Modal was chosen for the comparison, because the
article deadline matters more than provider parity and the Modal evidence is
already sufficient on its own.

## What exists in AWS

Nothing that bills. The key pair and the security group remain from a partial
apply, both free, and they exist only in the local git-ignored state file. There
is no instance, no Spot request, no volume, and no state lock.

## What is in this directory

`main.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars.example`, and the
generated `.terraform.lock.hcl`. One instance, one security group, one key pair,
reusing the default VPC. Spot is a market option on an ordinary instance, so
`terraform destroy` has the normal terminate path.

The runtime lives elsewhere on purpose: `aws/run_experiment.py` runs the
experiments, and Terraform only reproduces the machine.

## One correction worth keeping

The `timeouts { create = "30m" }` block does **not** bound a queued Spot
request. It was added on the assumption that `aws_instance`'s default 10 minute
create timeout cancels a queued request; the 47 minute run disproved that. The
time was the AWS SDK's own retry loop, reported as `exceeded maximum number of
attempts, 25`. The block is kept because a bounded create is still sensible, but
it is not the mechanism that governs a queued Spot request.

## To resume

The Terraform is valid and was verified with `fmt -check` and `validate` against
provider 6.66.0. The only missing input is an availability zone with capacity,
set locally as `subnet_id` in the git-ignored `terraform.tfvars`. Re-run
`terraform plan` and confirm it still reports 1 to add, 0 to change, 0 to
destroy before applying.
