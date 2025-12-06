import pandas as pd
import pm4py

#Wczytywania event logu
df = pd.read_csv('Event_log.csv', sep=',')


#Konwersja even logu na data frame zgodnie z wymogami PM4PY
df = pm4py.format_dataframe(df, case_id='Order_ID', activity_key='Activity', timestamp_key='Timestamp')
event_log = pm4py.convert_to_event_log(df)
print(f'{50 * "-"} Analiza wariantów {50 * "-"}')


#Sprawdzanie aktywności początkowej każdego zamówienia
print(f'\nWeryfikacja początkowej aktywności:')
start_activities = pm4py.get_start_activities(event_log, activity_key='Activity', case_id_key='Order_ID', timestamp_key='Timestamp')

for key, value in start_activities.items():
    print(f'{value} zamówień rozpoczyna się od aktywności "{key}".')

#Sprawdzanie jakimi aktywnościami kończy się każde zamówienie
print(f'\nWeryfikacja końcowej aktywności:')
end_activities = pm4py.get_end_activities(event_log, activity_key='Activity', case_id_key='Order_ID', timestamp_key='Timestamp')

for key, value in end_activities.items():
    print(f'{value} zamówień kończy się aktywnością "{key}"')

#Sprawdzanie ilosci wariantow
print(f'\n {50 * "-"} Sprawdzanie ilości wartiantów {50 * "-"}\n')
variants = pm4py.get_variants(event_log, activity_key='Activity', case_id_key='Order_ID', timestamp_key='Timestamp')
print(f"W Event logu jest {len(variants)} wariantów o następującej strukturze:\n")

#Sprawdzanie ilosci aktywnosci w wariancie
for i, (variant, subdf) in enumerate(pm4py.split_by_process_variant(df), start=1):
    print(f"Wariant {i} składający się z {len(subdf)} aktywności o strukturze: {variant}")

#Sprawdzanie ilosci zamowien w wariantach
print(f'\nDo poszczególnych wariantów należy następująca liczba zamówień:\n')

variant_counter = 1

for variant, subdf in pm4py.split_by_process_variant(df):
    variant_name = f"Wariant {variant_counter}"
    count_orders = subdf["Order_ID"].nunique()
    print(f"{variant_name}: {count_orders} zamówień")

    variant_counter += 1

