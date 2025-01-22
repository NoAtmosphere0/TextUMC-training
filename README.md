# Thesis-test-code

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/thesis-test-code.git
cd thesis-test-code

# Create virtual environment with micromamba (recommended)
micromamba create -n venv python==3.12.8
micromamba activate venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
- Normal (unified) training mode
```bash
python model2.py --train_type normal --num_epochs 10  # currently batch size is fixed at -1 for normal training
```

- Per-claim (demonic) training mode
```bash
python model2.py --train_type demonic --num_epochs 5 --batch_size 32 # dunno if batch_size works now
```

