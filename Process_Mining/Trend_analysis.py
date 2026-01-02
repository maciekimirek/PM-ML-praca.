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

print(f'\n{50 * "-"} Analiza trendów {50 * "-"}\n')
print(f'\n{50 * "-"}Szukanie trendów w rozkładzie czasu {50 * "-"}\n')

print("Przygotowywanie wykresu i wybieranie trendu do analizy")

pm4py.view_events_per_time_graph(
    event_log,
    activity_key='Activity',
    case_id_key='Order_ID',
    timestamp_key='Timestamp'
)

fig = plt.gcf()
fig.savefig("events_per_time.png", dpi=200, bbox_inches='tight')
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

print(f'\n{50 * "-"}Szukanie trendów w rozkładzie godzinnym {50 * "-"}\n')
print("Koniec")

pm4py.view_events_distribution_graph(event_log, distr_type='hours', activity_key='Activity', case_id_key='Order_ID', timestamp_key='Timestamp')
plt.show()

print("Koniec")
#
# pm4py.view_events_distribution_graph(event_log, distr_type='days_week', activity_key='Activity', case_id_key='Order_ID', timestamp_key='Timestamp')
# plt.show()