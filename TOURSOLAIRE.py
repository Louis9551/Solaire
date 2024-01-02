# Données
Pel = 538.56# kW Puissance électrique nominale
Ta = 25+273 # °K Température ambiante
Tf = 250+273 # °K Température de fonctionnement
Alpha_sc = 0.7 # Absorptivité du capteur
Epsilon_sc = 0.85 # Emissivité du capteur
S = 1000 # W/m² Irradiation solaire
Cg = 500 # Concentration solaire
sigma = 5.67*10**(-8) # Constante de Stefan-Boltzmann
Selio = 120 # m² surface d'un héliosta
PrixSeville = 35*10**6 # € Prix d'une centrale à Seville
NbElioSeville = 624 # Nombre d'héliostats à Seville

# Résulution 
eta_c = 2/3*(1-Ta/Tf)
print("etaC = ",eta_c)
eta_sc = Alpha_sc-(Epsilon_sc*sigma*(Tf**4-Ta**4))/(S*Cg)
print("etaSC = ",eta_sc)
Qsc = Pel/eta_sc/eta_c
print("Qsc = ",Qsc,"kW")
Stot = Qsc*1000/S
NbElio = round(Stot/Selio)
print("NbElio = ",NbElio)
Stot = NbElio*Selio
print("Stot = ",Stot,"m²")
Prix = PrixSeville*NbElio/NbElioSeville
print("Prix = ",Prix,"€")

