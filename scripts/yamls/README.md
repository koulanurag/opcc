## Usage

- Create AMLT Project
```bash
amlt project create <name-of-project> <storage-account-name> 
```

- Generate YAML
```bash
python generate_yaml.py --per-node-commands 10 --prefix-yaml-path prefix.yaml
```

- Submit Jobs
```bash
amlt run td3_bc.yaml td3_bc
```