import pandas as pd
import numpy as np

# Read original dataset
df = pd.read_csv("data/death_sex_ethnic_state.csv")

# Extract year from date
df["year"] = pd.to_datetime(df["date"]).dt.year

# Choose years for radar chart
selected_years = [2000, 2010, 2020, 2022]

# Keep national-level profile by summing all states
# Keep sex = both, remove overall because it is the total, not an ethnicity
radar = df[
    (df["sex"] == "both") &
    (df["ethnicity"] != "overall") &
    (df["year"].isin(selected_years))
].copy()

# Sum deaths across states for each year and ethnicity
radar = (
    radar
    .groupby(["year", "ethnicity"], as_index=False)["abs"]
    .sum()
)

# Calculate percentage share of deaths within each year
radar["total_deaths"] = radar.groupby("year")["abs"].transform("sum")
radar["share"] = radar["abs"] / radar["total_deaths"] * 100

# Set fixed radar axis order
ethnicity_order = [
    "bumi_malay",
    "bumi_other",
    "chinese",
    "indian",
    "other_citizen",
    "other_noncitizen"
]

order_map = {ethnicity: i for i, ethnicity in enumerate(ethnicity_order)}
radar["order"] = radar["ethnicity"].map(order_map)

# Calculate radar coordinates
n = len(ethnicity_order)
radar["angle"] = 2 * np.pi * radar["order"] / n
radar["radius"] = radar["share"] / 50
radar["radius"] = radar["radius"].clip(upper=1)

radar["x"] = radar["radius"] * np.sin(radar["angle"])
radar["y"] = -radar["radius"] * np.cos(radar["angle"])

# Duplicate first point of each year so the radar line closes
closed_rows = []

for year in selected_years:
    year_data = radar[radar["year"] == year].sort_values("order")
    closed_rows.append(year_data)

    first_row = year_data.iloc[0].copy()
    first_row["order"] = n
    first_row["angle"] = 2 * np.pi
    first_row["x"] = first_row["radius"] * np.sin(first_row["angle"])
    first_row["y"] = -first_row["radius"] * np.cos(first_row["angle"])
    closed_rows.append(pd.DataFrame([first_row]))

radar_closed = pd.concat(closed_rows, ignore_index=True)

# Save main radar data
radar_closed.to_csv("data/death_ethnicity_radar.csv", index=False)

# Create radar grid rings
grid_rows = []
rings = [25, 50, 75, 100]

for ring in rings:
    radius = ring / 100
    for i in range(n + 1):
        angle = 2 * np.pi * (i % n) / n
        grid_rows.append({
            "ring": ring,
            "order": i,
            "x": radius * np.sin(angle),
            "y": -radius * np.cos(angle)
        })

grid = pd.DataFrame(grid_rows)
grid.to_csv("data/death_radar_grid.csv", index=False)

# Create axis lines and labels
axis_rows = []

label_names = {
    "bumi_malay": "Bumi Malay",
    "bumi_other": "Bumi Other",
    "chinese": "Chinese",
    "indian": "Indian",
    "other_citizen": "Other Citizen",
    "other_noncitizen": "Other Non-Citizen"
}

for ethnicity in ethnicity_order:
    i = order_map[ethnicity]
    angle = 2 * np.pi * i / n

    axis_rows.append({
        "ethnicity": ethnicity,
        "label": label_names[ethnicity],
        "x": 0,
        "y": 0,
        "x2": np.sin(angle),
        "y2": -np.cos(angle),
        "label_x": 1.16 * np.sin(angle),
        "label_y": -1.16 * np.cos(angle)
    })

axis = pd.DataFrame(axis_rows)
axis.to_csv("data/death_radar_axes.csv", index=False)

print("Radar CSV files created successfully.")