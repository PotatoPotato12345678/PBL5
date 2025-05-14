### how to set up the environment in Ubuntu (Linux)

install brew
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
run the command suggested in the output and
check if correcly installed
```
brew --version
```

Install uv
```
brew install uv
```

install the environment from pyproject.toml
```
uv sync
```

activate the virtual environment
```
. .venv/bin/activate
```

run the jupyter lab
```
jupyter lab
```

### IP102 for classification
https://drive.google.com/file/d/1EL9TA-J5XsiBR4M3nQkMR_e-2dP48eQN/view?usp=drive_link

