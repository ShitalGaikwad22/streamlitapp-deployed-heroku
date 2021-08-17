mkdir -p ~/.streamlit/
echo "
[general]n
email = "shitalgaikwad9097@gmail.com"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml
