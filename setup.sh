mkdir -p ~/streamlit

echo "\
[server]\n\
port = 8501\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/streamlit/config.toml
