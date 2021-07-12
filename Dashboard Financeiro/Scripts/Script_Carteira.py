# Importando as bibliotecas
import pandas as pd
import pandas_datareader as wb
import numpy as np
import statsmodels.api as sm

"""# Montando a Carteira"""

# Importando movimentações da carteira
movimentacao_carteira = pd.read_excel('D:/Google Drive/Colab Notebooks/Portfólio/Dashboard Financeiro/Datasets/movimentacoes_carteira.xls')
movimentacao_carteira.dropna(inplace=True)

# Calculando o preço médio das ações
def calcula_preco_medio(compras, vendas):
  compras['VALOR PAGO'] = compras['PRECO'] * compras['VOLUME'] + compras['TAXAS']
  vendas['VALOR PAGO'] = (vendas['PRECO'] * vendas['VOLUME'])* -1 + vendas['TAXAS']
  volume = compras['VOLUME'].sum() - vendas['VOLUME'].sum()
  if volume > 0:
    preco_medio = np.round(((compras['VALOR PAGO'].sum() + vendas['VALOR PAGO'].sum()) / volume), decimals=2)
  else:
    preco_medio = 0
  return preco_medio, volume

# Montando a carteira
carteira = []
for t in np.sort(movimentacao_carteira['CODIGO'].unique()):
  acao = movimentacao_carteira.loc[movimentacao_carteira['CODIGO']==t]
  preco_medio, volume = calcula_preco_medio(acao.loc[acao['OPERACAO']=='Compra'], acao.loc[acao['OPERACAO']=='Venda'])
  if volume > 0:
    carteira.append([t, preco_medio, volume])

# Convertendo para um DataFrame    
carteira = pd.DataFrame(data=carteira, columns=['Papel', 'Preco_Medio', 'Volume'])

# Calculando pesos dos ativos
carteira['Pesos'] =carteira['Volume'] / carteira['Volume'].sum()
print(carteira)

"""# Ativos"""

def importar_ativos(ativos, data_inicio = None, data_fim = None):
  # Criando o dataframe e preenchendo com dados do primeiro ativos
  df_ativos = pd.DataFrame()
  df_ativos = wb.DataReader(ativos[0], data_source='yahoo', start=data_inicio, end=data_fim)
  df_ativos.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume'], axis=1, inplace=True)
  if ativos[0][0] == '^':
    df_ativos.rename({"Adj Close":str(ativos[0])}, axis=1, inplace=True)
  else:
    df_ativos.rename({"Adj Close":str(ativos[0][:-3])}, axis=1, inplace=True)
  df_ativos['Data'] = df_ativos.index
  # Completando o dataframe com dados dos outros ativos
  for i in ativos[1:]:
    df = wb.DataReader(i, data_source='yahoo', start=data_inicio, end=data_fim)
    df.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume'], axis=1, inplace=True)
    if i[0] == '^':
      df_ativos.rename({"Adj Close":str(i)}, axis=1, inplace=True)
    else:
      df_ativos.rename({"Adj Close":str(i[:-3])}, axis=1, inplace=True)
    df.rename({"Adj Close":str(i[:-3])}, axis=1, inplace=True)
    df['Data'] = df.index
    df_ativos = df_ativos.merge(df, how='inner', on='Data')
  # Retornando resultado
  return df_ativos

cotacoes_hist_ativos = importar_ativos(carteira['Papel'] + '.SA', '2019-01-01')
print(cotacoes_hist_ativos)

cotacoes_hist_ibov = importar_ativos(['^BVSP'], '2019-01-01')
print(cotacoes_hist_ibov)

def calcular_retornos(df_ativos, ativos):
  df_retornos = df_ativos[ativos].pct_change()
  df_retornos['Data'] = df_ativos['Data']
  df_retornos.dropna(inplace=True)
  return df_retornos

retornos_ativos = calcular_retornos(cotacoes_hist_ativos, carteira['Papel'])
print(retornos_ativos)
retornos_ibov = calcular_retornos(cotacoes_hist_ibov, ['^BVSP'])

# Retorno Acumulado dos Ativos
def calcular_retorno_acumulado(df_retornos, ativos):
  retorno_acumulado = (1 + df_retornos[ativos]).cumprod()
  retorno_acumulado['Data'] = df_retornos['Data']
  return retorno_acumulado

retorno_ativos_acum = calcular_retorno_acumulado(retornos_ativos, carteira['Papel'])
print(retorno_ativos_acum)

# Volatilidade dos ativos
volatilidade_ativos = retornos_ativos.std()
volatilidade_ativos = pd.DataFrame(data = volatilidade_ativos[carteira['Papel']], columns=['Volatilidade'])
volatilidade_ativos['Papel'] = volatilidade_ativos.index
volatilidade_ativos.reset_index(inplace=True, drop=True)
print(volatilidade_ativos)

volatilidade_ibov = retornos_ibov.std()
volatilidade_ibov = pd.DataFrame(data = [volatilidade_ibov['^BVSP']], columns=['Volatilidade'])
volatilidade_ibov['Papel'] = '^BVSP'
volatilidade_ativos.reset_index(inplace=True, drop=True)
print(volatilidade_ibov)

# Value-at-Risk
def calcular_var_historico(df_retornos, ativos, conf):
  var = []
  for i in ativos:
    var.append([i, np.nanpercentile(df_retornos[i], conf)])
  return var

var_hist_ativos_90 = pd.DataFrame(data = calcular_var_historico(retornos_ativos, carteira['Papel'], 10), columns=['Papel', 'VaR'])
print(var_hist_ativos_90)
var_hist_ativos_95 = pd.DataFrame(data = calcular_var_historico(retornos_ativos, carteira['Papel'], 5), columns=['Papel', 'VaR'])
print(var_hist_ativos_95)
var_hist_ativos_99 = pd.DataFrame(data = calcular_var_historico(retornos_ativos, carteira['Papel'], 1), columns=['Papel', 'VaR'])
print(var_hist_ativos_99)

# Value-at-Risk Parametrico
from scipy.stats import norm
def calcular_var_parametrico(df_retornos, ativos, conf):
  var = []
  for i in ativos:
    media_retorno = np.mean(df_retornos[i])
    desvio_padrao = np.std(df_retornos[i])
    var_par = norm.ppf(conf/100, media_retorno, desvio_padrao)
    var.append([i, var_par])
  return var

var_par_ativos_90 = pd.DataFrame(data = calcular_var_parametrico(retornos_ativos, carteira['Papel'], 10), columns=['Papel', 'VaR'])
print(var_par_ativos_90)
var_par_ativos_95 = pd.DataFrame(data = calcular_var_parametrico(retornos_ativos, carteira['Papel'], 5), columns=['Papel', 'VaR'])
print(var_par_ativos_95)
var_par_ativos_99 = pd.DataFrame(data = calcular_var_parametrico(retornos_ativos, carteira['Papel'], 1), columns=['Papel', 'VaR'])
print(var_par_ativos_99)

# Beta por Regressão Linear
def calcular_beta_regressao(df_retornos, ativos):
  # Construindo variáveis X (independente) e Y (dependente) e calculando Beta
  X = df_retornos['^BVSP']
  X = sm.add_constant(X)
  beta = []
  for i in ativos:
    Y = df_retornos[i]
    modelo = sm.OLS(Y, X)
    resultado = modelo.fit()
    beta.append([i , resultado.params['^BVSP']])
  return beta

beta_rl_ativos = pd.DataFrame(data = calcular_beta_regressao(retornos_ibov.merge(retornos_ativos, how='inner', on = 'Data'), 
                            carteira['Papel']), columns = ['Papel', 'Beta']) 
print(beta_rl_ativos)

"""# Carteira"""

# Retorno da Carteira
def calcular_retorno_carteira(df_retornos, ativos, pesos):
  retorno_carteira = pd.DataFrame(columns=['Retorno'])
  retorno_carteira['Retorno'] = (df_retornos[ativos] * pesos).sum(axis=1)
  retorno_carteira['Data'] = df_retornos['Data']
  return retorno_carteira

retorno_carteira = calcular_retorno_carteira(retornos_ativos, carteira['Papel'], carteira['Pesos'].values)
print(retorno_carteira)

# Retorno Acumulado da Carteira
def calcular_retorno_acumulado_carteira(df_retornos, ativos, pesos):
  retorno_acumulado = calcular_retorno_acumulado(df_retornos, ativos)
  retorno_acum_carteira = pd.DataFrame(columns=['Carteira'])
  retorno_acum_carteira['Carteira'] = (retorno_acumulado[ativos] * pesos).sum(axis=1)
  retorno_acum_carteira['Data'] = df_retornos['Data']
  return retorno_acum_carteira

retorno_acumulado_carteira = calcular_retorno_acumulado_carteira(retornos_ativos, carteira['Papel'], carteira['Pesos'].values)
print(retorno_acumulado_carteira)

# Ibov Acumulado
retornos_ibov_acum = calcular_retorno_acumulado(retornos_ibov, ['^BVSP'])
print(retornos_ibov_acum)

# Volatilidade da Carteira
def calcular_volatilidade_carteira(df_retornos, pesos):
  #1º calcular a matriz de covariância
  matriz_cov = df_retornos.cov()
  #Calcular a volatilidade
  vol_carteira = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))
  #Vol Carteira Ano
  vol_carteira_ano = vol_carteira*np.sqrt(252)
  return vol_carteira, vol_carteira_ano

volatilidade_carteira = pd.DataFrame(data = [calcular_volatilidade_carteira(retornos_ativos, carteira['Pesos'].values)], columns = ['Carteira', 'Ano']) 
print(volatilidade_carteira)

# Beta Carteira
beta_carteira = pd.DataFrame(data = calcular_beta_regressao(retornos_ibov.merge(retorno_carteira, how='inner', on='Data'), ['Retorno']), columns = ['Papel', 'Beta']) 
print(beta_carteira)

# VaR Histórico da Carteira 
var_hist_carteira_90 = pd.DataFrame(data = calcular_var_historico(retorno_carteira, ['Retorno'], 10), columns=['Papel', 'VaR'])
print(var_hist_carteira_90)
var_hist_carteira_95 = pd.DataFrame(data = calcular_var_historico(retorno_carteira, ['Retorno'], 5), columns=['Papel', 'VaR'])
print(var_hist_carteira_95)
var_hist_carteira_99 = pd.DataFrame(data = calcular_var_historico(retorno_carteira, ['Retorno'], 1), columns=['Papel', 'VaR'])
print(var_hist_carteira_99)

# VaR Paramétrico da Carteira 
var_par_carteira_90 = pd.DataFrame(data = calcular_var_parametrico(retorno_carteira, ['Retorno'], 10), columns=['Papel', 'VaR'])
print(var_par_carteira_90)
var_par_carteira_95 = pd.DataFrame(data = calcular_var_parametrico(retorno_carteira, ['Retorno'], 5), columns=['Papel', 'VaR'])
print(var_par_carteira_95)
var_par_carteira_99 = pd.DataFrame(data = calcular_var_parametrico(retorno_carteira, ['Retorno'], 1), columns=['Papel', 'VaR'])
print(var_par_carteira_99)

"""# PyPortfolioOpt"""

carteira_passado = importar_ativos(carteira['Papel'] + '.SA', '2016-01-01', '2018-12-31')
carteira_futuro = importar_ativos(carteira['Papel'] + '.SA', '2019-01-01', '2020-12-31')

# 2º Chamando Ibovespa
ibov_in = importar_ativos(['^BVSP'], '2016-01-01', '2018-12-31')
ibov_out = importar_ativos(['^BVSP'], '2019-01-01', '2020-12-31')
ibov_in.drop(columns=['Data'], inplace=True)
ibov_out.drop(columns=['Data'], inplace=True)

from pypfopt import expected_returns

# Retorno Médio Out of Sample
retorno_medio_f = expected_returns.mean_historical_return(carteira_futuro[carteira['Papel']])
print(retorno_medio_f)

# Retorno Médio In Sample
retorno_medio = expected_returns.mean_historical_return(carteira_passado[carteira['Papel']])
print(retorno_medio)

#Erro Médio Absoluto (Mean Absolut Error)
#|retorno_passado - retorno_futuro|/(n_retorno_passado)
ema_retorno_medio = np.sum(np.abs(retorno_medio-retorno_medio_f))/len(retorno_medio)

print(ema_retorno_medio)

#Media Móvel Exponencial - Out Of Sample
mme_f = expected_returns.ema_historical_return(carteira_futuro[carteira['Papel']], span=700)
print(mme_f)

#Media Móvel Exponencial - In Sample
mme = expected_returns.ema_historical_return(carteira_passado[carteira['Papel']], span=700)
print(mme)

#Erro Médio Absoluto MME
ema_mme = np.sum(np.abs(mme-mme_f))/len(mme)
print(ema_mme)

# 1º calcular a selic diária
selic_aa = 0.0265
selic_diaria = (1+selic_aa)**(1/252) -1
selic_diaria

# CAPM Out of Sample
capm_f = expected_returns.capm_return(carteira_futuro[carteira['Papel']].set_index(carteira_futuro['Data']), market_prices=ibov_out, risk_free_rate=selic_diaria)
capm_f = pd.DataFrame(data=capm_f.values, columns=['CAPM'])
capm_f['Papel'] = carteira['Papel']
capm_f.reset_index(inplace=True, drop=True)
print(capm_f)

# CAPM in sample
capm = expected_returns.capm_return(carteira_passado[carteira['Papel']].set_index(carteira_passado['Data']), market_prices=ibov_in, risk_free_rate=selic_diaria)
capm = pd.DataFrame(data=capm.values, columns=['CAPM'])
capm['Papel'] = carteira['Papel']
capm.reset_index(inplace=True, drop=True)
print(capm)

# Erro Médio Absoluto CAPM
mae_capm = np.sum(np.abs(capm['CAPM']-capm_f['CAPM']))/len(capm['CAPM'])
print(mae_capm)

"""# Modelos de Risco"""

# Importando o método risk_models
from pypfopt import risk_models

#Sample Cov out-of-sample
sample_cov_f = risk_models.sample_cov(carteira_futuro[carteira['Papel']])
print(sample_cov_f)

# Sample Cov in-sample
sample_cov = risk_models.sample_cov(carteira_passado[carteira['Papel']])
print(sample_cov)

# Mean Absolut Error
mae_sample_cov = np.sum(np.abs(np.diag(sample_cov)-np.diag(sample_cov_f))/len(np.diag(sample_cov)))
print(mae_sample_cov)

#Semicovariance out of sample
semicov_f = risk_models.semicovariance(carteira_futuro[carteira['Papel']], benchmark=0.02)
print(semicov_f)

#Semicovariance in sample
semicov = risk_models.semicovariance(carteira_passado[carteira['Papel']], benchmark=0.02)
print(semicov)

# Mean Absolut Error Semicovariance
mae_semicov = np.sum(np.abs(np.diag(semicov)-np.diag(semicov_f))/len(np.diag(semicov)))
print(mae_semicov)

# EWC Out of Sample
exp_cov_f = risk_models.exp_cov(carteira_futuro[carteira['Papel']])
print(exp_cov_f)

# EWC In Sample
exp_cov = risk_models.exp_cov(carteira_passado[carteira['Papel']])
print(exp_cov)

# MAE Expo_Cov
mae_expcov = np.sum(np.abs(np.diag(exp_cov)-np.diag(exp_cov_f))/len(np.diag(exp_cov)))
print(mae_expcov)

#Retorno Anualizado
cf_anualizado = ((carteira_futuro[carteira['Papel']].iloc[-1]-carteira_futuro[carteira['Papel']].iloc[0])/carteira_futuro[carteira['Papel']].iloc[0])
cf_anualizado = ((1+cf_anualizado)**(12/24))-1
cf_anualizado = pd.DataFrame(data=cf_anualizado, columns=['Retorno Anualizado'])
cf_anualizado['Papel'] = cf_anualizado.index
cf_anualizado.reset_index(inplace=True, drop=True)
print(cf_anualizado)

#Retorno Carteira = Pesos x retornos individuais
fut_ret = cf_anualizado['Retorno Anualizado'].dot(carteira['Pesos'].values)
fut_ret = pd.DataFrame(data=[fut_ret], columns=['Retorno Anualizado Carteira'])
print(fut_ret)

#Volatilidade anualizada
retorno_cf = carteira_futuro[carteira['Papel']].pct_change()
fut_cov = retorno_cf.cov()
fut_vol = np.sqrt(np.dot(carteira['Pesos'].values.T, np.dot(fut_cov,carteira['Pesos'].values)))
vol_ano = fut_vol*np.sqrt(252)
vol_ano = pd.DataFrame(data=[vol_ano], columns=['Volatilidade Anualizada'])
print(vol_ano)

#Anualizar a matriz de covariância
fut_cov_anual = fut_cov*252

#Retornos Ibovespa
ibov_pos = wb.DataReader('^BVSP', data_source='yahoo', start='2019-01-01', end='2020-12-31')
ibov_pos = ibov_pos['Adj Close']
ibov_pos_retornos = ibov_pos.pct_change()
ibov_anualizado = ((ibov_pos.iloc[-1]-ibov_pos.iloc[0])/ibov_pos.iloc[0])
ibov_anualizado = ((1+ibov_anualizado)**(12/24))-1
ibov_anualizado = pd.DataFrame(data=[ibov_anualizado], columns=['Ibov Anualizado'])
print(ibov_anualizado)

ibov_vol = ibov_pos_retornos.std()
ibov_vol_ano = ibov_vol*np.sqrt(252)
ibov_vol_ano = pd.DataFrame(data=[ibov_vol_ano], columns=['Ibov Vol Ano'])
print(ibov_vol_ano)

#Importar Modulo Expected Returns
from pypfopt import expected_returns

re = expected_returns.capm_return(carteira_passado[carteira['Papel']].set_index(carteira_passado['Data']), market_prices=ibov_in, risk_free_rate=selic_diaria)
re = pd.DataFrame(data=re.values, columns=['Retorno Esperado'])
re['Papel'] = carteira['Papel']
re.reset_index(inplace=True, drop=True)
print(re)

#Diferença entre retornos estimados e retornos reais
dif_retorno = re['Retorno Esperado'] - cf_anualizado['Retorno Anualizado']

# Importar módulo de otimização Fronteira Eficiente
from pypfopt import EfficientFrontier

mv = EfficientFrontier(re['Retorno Esperado'], sample_cov)

# Mínima Volatilidade
mv.min_volatility()
pesos_min_vol = mv.clean_weights()
pesos_min_vol = pd.DataFrame(data=pd.Series(pesos_min_vol), columns=['Peso'])
pesos_min_vol['Papel'] = carteira['Papel']
print(pesos_min_vol)

print(mv.portfolio_performance(verbose=True, risk_free_rate=selic_aa))

vol_otimizada = np.sqrt(np.dot(pesos_min_vol['Peso'].T, np.dot(fut_cov,pesos_min_vol['Peso'])))
vol_otimizada = pd.DataFrame(data=[vol_otimizada*np.sqrt(252)], columns=['Vol_Otim'])
print(vol_otimizada)

#Importar o módulo funçao Regularizadora
from pypfopt import objective_functions

mv_2 = EfficientFrontier(re['Retorno Esperado'], sample_cov)
mv_2.add_objective(objective_functions.L2_reg, gamma=0.1)
mv_2.min_volatility()
pesos_min_vol_reg = mv_2.clean_weights()
pesos_min_vol_reg = pd.DataFrame(data=pd.Series(pesos_min_vol_reg), columns=['Peso'])
pesos_min_vol_reg['Papel'] = carteira['Papel']
print(pesos_min_vol_reg)

msharpe = EfficientFrontier(re['Retorno Esperado'],sample_cov)
msharpe.max_sharpe(risk_free_rate=selic_aa)
sharpe_pesos = msharpe.clean_weights()
sharpe_pesos = pd.DataFrame(data=pd.Series(sharpe_pesos), columns=['Peso'])
sharpe_pesos['Papel'] = carteira['Papel']
print(sharpe_pesos)

print(msharpe.portfolio_performance(verbose=True, risk_free_rate=selic_aa))

