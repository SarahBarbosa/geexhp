## Habitable Worlds Observatory (HWO): Gerador de Espectros de Reflexão de Exoplanetas Tipo Terra com PSG

(Documentação em Desenvolvimento)

# Instruções de Instalação

## 1. Clonar o Repositório

```bash
git clone <URL_DO_REPOSITÓRIO>
```

## 2. Criar um Ambiente Virtual (Linux)

```bash
python3 -m venv psg-venv
source psg-venv/bin/activate
```

## 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

## 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto e adicione as seguintes linhas:

```plaintext
habex_config=/CAMINHO/PARA/scr/config/default_habex.config
funcoes_path=/CAMINHO/PARA/scr
```

Substitua `/CAMINHO/PARA/` pelo caminho real onde os arquivos estão localizados.
