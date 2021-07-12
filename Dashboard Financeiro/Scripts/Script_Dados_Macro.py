# Importando as bibliotecas
import pandas as pd
import numpy as np
import quandl

# Configurando o acesso a Quandl
quandl.api_config.read_key('D:/Google Drive/Colab Notebooks/Portfólio/Dashboard Financeiro/Quandl_Key.txt')

# PIB 
# - Fonte: FMI
# - Frequência: Anual
# - Unidade de Medida: Bilhões de U$D
pib = quandl.get('ODA/BRA_NGDPD', start_date = '2010-01-01')
pib.rename (columns={'Value':'PIB'}, inplace=True)
pib['Data'] = pib.index

# IPCA
# - Fonte: BCB
# - Frequência: Mensal
# - Unidade de Medida: Percentual (%)
ipca = quandl.get('BCB/13522', start_date = '2010-01-01')
ipca.rename (columns={'Value':'IPCA'}, inplace=True)
ipca['Data'] = ipca.index

# Selic 
# - Fonte: BCB
# - Frequência: Diária
# - Unidade de Medida: Percentual (%)
selic = quandl.get('BCB/432', start_date = '2010-01-01')
selic.rename (columns={'Value':'Selic'}, inplace=True)
selic['Data'] = selic.index

# IGPM 
# - Fonte: BCB
# - Frequência: Mensal
# - Unidade de Medida: Percentual (%)
igpm = quandl.get('BCB/189', start_date = '2010-01-01')
igpm.rename (columns={'Value':'IGPM'}, inplace=True)
igpm['Data'] = igpm.index

# INPC 
# - Fonte: BCB
# - Frequência: Mensal
# - Unidade de Medida: Percentual (%)
inpc = quandl.get('BCB/188', start_date = '2010-01-01')
inpc.rename (columns={'Value':'INPC'}, inplace=True)
inpc['Data'] = inpc.index

# PIM-PF 
# - Fonte: IBGE
# - Frequência: Mensal
# - Unidade de Medida: Percentual (%)
# - Base: média de 2012 = 100 %
pimpf = pd.read_csv('D:/Google Drive/Colab Notebooks/Portfólio/Dashboard Financeiro/Datasets/pim-pf.csv', sep=';', encoding='UTF-8')
pimpf.drop(columns = ['Variável', 'Unnamed: 2', 'Seções e atividades industriais (CNAE 2.0)'], axis =1, inplace=True)
pimpf.rename(columns={'Unnamed: 4':'PIM-PF'}, inplace=True)

# Câmbio (Dolar) 
# - Fonte: BCB
# - Frequência: Diária
# - Unidade de Medida: R$
dolar = quandl.get('BCB/10813', start_date = '2010-01-01')
dolar.rename (columns={'Value':'Dolar'}, inplace=True)
dolar['Data'] = dolar.index

# Desemprego 
# - Fonte: BCB
# - Frequência: Mensal
# - Unidade de Medida: %
desemprego = quandl.get('BCB/24369', start_date =  '2010-01-01')
desemprego.rename (columns={'Value':'Desemprego'}, inplace=True)
desemprego['Data'] = desemprego.index

# Resultados
print(pib)
print(ipca)
print(selic)
print(igpm)
print(inpc)
print(pimpf)
print(dolar)
print(desemprego)