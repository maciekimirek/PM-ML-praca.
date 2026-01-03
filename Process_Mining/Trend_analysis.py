import pm4py
import pandas as pd
import matplotlib.pyplot as plt

# Wczytywanie event logu
df = pd.read_csv('../Data_source/Event_log.csv', sep=',')

# Konwersja even logu na data frame zgodnie z wymogami PM4PY
df = pm4py.format_dataframe(
    df,
    case_id='Order_ID',
    activity_key='Activity',
    timestamp_key='Timestamp'
)
event_log = pm4py.convert_to_event_log(df)

# 1. Analiza na podstawie dat

print(f'\n{50 * "-"} Analiza trendów {50 * "-"}\n')
print(f'\n{41 * "-"}Szukanie trendów w rozkładzie czasu {40* "-"}\n')

print("Przygotowywanie wykresu i wybieranie trendu do analizy")

pm4py.view_events_per_time_graph(
    event_log,
    activity_key='Activity',
    case_id_key='Order_ID',
    timestamp_key='Timestamp'
)

fig = plt.gcf()
fig.savefig("events_per_date.png", dpi=200, bbox_inches='tight')
plt.close(fig)



# Filtrowanie DF na podstawie wykresu aby przeprowadzic analize

print("Filtrowanie zakresu dat w Event Logu na podstawie wykresu: 2024-04-10 → 2024-04-20")

df_filtered = df[
    (df["Timestamp"] >= "2024-04-10") &
    (df["Timestamp"] <= "2024-04-20")
]

print(f"Liczba eventów w tym okresie: {len(df_filtered)}")
print(f"Liczba unikalnych case'ów: {df_filtered['Order_ID'].nunique()}\n")


# TOP Aktywnsci
print("Dominujące aktywności w okresie piku:\n")

print(df_filtered["Activity"].value_counts(), "\n")


# Liczba eventow per day
print("Liczba eventów dziennie w okresie piku:\n")

events_per_day = df_filtered.groupby(df_filtered["Timestamp"].dt.date).size()
print(events_per_day, "\n")


# Porownanie ze stabilnym okresem
print("Porównanie struktury aktywności:")
print("Okres referencyjny: 2024-03-10 → 2024-03-20\n")

df_normal = df[
    (df["Timestamp"] >= "2024-03-10") &
    (df["Timestamp"] <= "2024-03-20")
]

diff = (
    df_filtered["Activity"].value_counts(normalize=True)
    - df_normal["Activity"].value_counts(normalize=True)
).fillna(0).sort_values(ascending=False)

print("Zmiany udziału aktywności (w punktach procentowych):\n")
print((diff * 100).round(2))

#2. Analiza godzinna

print(f'\n{40 * "-"}Szukanie trendów w rozkładzie godzinnym {40 * "-"}\n')


pm4py.view_events_distribution_graph(
    event_log,
    distr_type='hours',
    activity_key='Activity',
    case_id_key='Order_ID',
    timestamp_key='Timestamp')

fig = plt.gcf()
fig.savefig("events_per_day.png", dpi=200, bbox_inches='tight')
plt.close(fig)


print("Na podstawie wygenerowanego wykresu (events_per_time) widzimy peak aktywności od godziniy 19 do 21.")
print(f"W tym przedziale czasowym są dokonywane nastepujace aktywnosci: \n")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df_filtered_by_hour = df[ (df["Timestamp"].dt.hour >= 19) & (df["Timestamp"].dt.hour < 21) ]
print(df_filtered_by_hour["Activity"].value_counts(), "\n")


print("Na podstawie powyższych liczb możemy wnioskować, że w godzinach wieczornych ludzie dokonują zamówien zdalnie po powrocie z pracy  ")
print(f'Zweryfikujmy to: \n')


orders_by_source = df.groupby("Order_Source")["Order_ID"].nunique()

percentage = []

for source in orders_by_source:
    result = (source/orders_by_source.sum())
    percentage.append((result * 100).round(2))

df_result = orders_by_source.reset_index(name="Orders")
df_result["Percentage share"] = percentage
df_result = df_result.set_index("Order_Source")
total_phone_online = df_result.loc[["Telefon", "Online"], "Percentage share"].sum()
print(df_result)
print(f'\nWidzimy, że udział procentowy zamówień złożonych zdalnie to {total_phone_online:.2f}% co może potwierdzać naszą hipoteze ')

#2. Analiza na podstawie dni tygodnia

print(f'\n{40 * "-"}Szukanie trendów w rozkładzie dni tygodnia {40 * "-"}\n')

pm4py.view_events_distribution_graph(
    event_log,
    distr_type='days_week',
    activity_key='Activity',
    case_id_key='Order_ID',
    timestamp_key='Timestamp')

fig1 = plt.gcf()
fig1.savefig("events_per_days_week.png", dpi=200, bbox_inches='tight')
plt.close(fig1)

print("W przypadku rozkladu aktywności na przestrzeni całego tygodnia nie widzmy żadnych poważnych odchyleń")
print("Zamówienia są sukcesywnie realizowane na przestrzeni całego tygodnia")
print(f"Sprawdźmy natomias czy samo składanie zamówienia dominuje w którymś dniu:\n")

df["day_of_week"] = df["Timestamp"].dt.dayofweek

df_filtered = df[df["Activity"] == "Złożenie zamówienia"]

df_day_of_week = df_filtered.groupby("day_of_week")["Order_ID"].nunique()

print(df_day_of_week)

print("Jak widzimy, zamówienia są składane równomiernie na przestrzeni całego tygodnia")



