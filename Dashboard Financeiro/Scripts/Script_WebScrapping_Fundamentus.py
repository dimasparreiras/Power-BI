# Importando as bibliotecas
import pandas as pd
import requests

# Informações para fingir ser um navegador
header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

# Função para preencher a URL do site com a ação a ser pesquisada
def Define_URL(codigo_acao):
    # URL do site
    url = "https://www.fundamentus.com.br/detalhes.php?papel=" + codigo_acao
    # Juntando a URL com as informações de navegador
    url = requests.get(url, headers=header)
    return url
  
# Função para ler e importar os dados do site Fundamentus
def Ler_URL(url):
# Lendo as informações do site Fundamentus
    acao = pd.read_html(url.text, 
                      decimal=",",
                      thousands=".")
    return acao

# Transpondo os valores da matriz
def Transpor_Matriz(acao):
    acao[0] = acao[0].transpose()
    acao[1] = acao[1].transpose()
    acao[2] = acao[2].iloc[1:].transpose()
    acao[3] = acao[3].iloc[1:].transpose()
    return acao

# Quebrando as linhas por grupo de informações e colocando todas em uma linha
def Quebrar_Informacoes(acao):
    # Quebrando as linhas
    info1 = acao[0].iloc[:2][:]
    info2 = acao[0].iloc[2:][:]
    info3 = acao[1].iloc[:2][:]
    info4 = acao[1].iloc[2:][:]
    info5 = acao[2].iloc[2:4][:]
    info6 = acao[2].iloc[4:][:]
    info7 = acao[3].iloc[:2][:]
    info8 = acao[3].iloc[2:][:]

    # Resetando os index
    info1.reset_index(inplace=True, drop=True)
    info2.reset_index(inplace=True, drop=True)
    info3.reset_index(inplace=True, drop=True)
    info4.reset_index(inplace=True, drop=True)
    info5.reset_index(inplace=True, drop=True)
    info6.reset_index(inplace=True, drop=True)
    info7.reset_index(inplace=True, drop=True)
    info8.reset_index(inplace=True, drop=True)

    # Concatenando as informações
    df_acao = pd.concat([info1, info2, info3, info4, info5, info6, info7, info8], # Tabela para fazer o merge
                      axis=1, # Unir pelas colunas
                      join='inner') # inner = o que é comum aos dois 
    return df_acao

# Retirando a interrogação do nome da coluna e a colocando como cabeçalho do DataFrame
def Corrigir_Nome_Colunas(df_acao):
    # Tirando a interrogação do nome da coluna
    colunas = df_acao.iloc[0][:]
    colunas.reset_index(drop = True, inplace = True)
    for i in range(len(colunas)):
      colunas[i] = colunas[i][1:]
    # Colocando como cabeçalho
    df_acao.columns = colunas
    df_acao.drop(0, axis=0, inplace=True)
    return df_acao

#Coletando os códigos de todas as ações listadas na B3  
acoes_ibov = pd.read_excel('D:/Google Drive/Colab Notebooks/Portfólio/Dashboard Financeiro/Datasets/papeis_B3.xlsx')

# Lendo os dados para cada uma das ações do IBOV
df_acao = pd.DataFrame()
for codigo_acao in acoes_ibov['Papel']:
    url = Define_URL(codigo_acao)
    acao = Ler_URL(url)
    if len(acao) >= 4:
        acao = Transpor_Matriz(acao)
        acao = Quebrar_Informacoes(acao)
        acao = Corrigir_Nome_Colunas(acao)
        df_acao = pd.concat([df_acao, acao], # Tabela para fazer o merge
                            axis=0, # Unir pelas linhas
                            join='outer') # outer = tudo dos dois 
    print('Dados da ação ' + codigo_acao + ' importados.')
                        
# Resetando o índice do DataFrame
df_acao.reset_index(inplace=True, drop=True)

# Tratando dados faltantes
df_acao.replace(to_replace='-', value='0', inplace=True)
df_acao.fillna(0, inplace=True)

# Exibindo o resultado
df_acao.to_csv('D:/Google Drive/Colab Notebooks/Portfólio/Dashboard Financeiro/Datasets/acoes_b3.csv')

print('Script executado com sucesso!!!')