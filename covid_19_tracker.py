import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import argparse
import sys


def setup_environment():
    """Set up the analysis environment and styling"""
    print("🔍 Starting COVID-19 Data Analysis...")
    
    # Make plots look nice
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    
    # Create output directory
    output_dir = './covid_charts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def load_data(file_path):
    """Load and validate the COVID-19 data from CSV"""
    print(f"\n📊 Loading data from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        print("   Please make sure the file exists in your current folder.")
        sys.exit(1)
    
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded data")
        print(f"   Found {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Validate essential columns
        required_columns = ['location', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Missing essential columns: {', '.join(missing_columns)}")
            sys.exit(1)
            
        return df
    
    except pd.errors.EmptyDataError:
        print("❌ Error: The file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("❌ Error: Unable to parse the CSV file. Please check the file format.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def explore_data(df):
    """Perform basic data exploration"""
    print("\n🔎 Exploring data...")
    
    # Check and convert date column
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            print(f"📅 Time period: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            print(f"⚠️ Warning: Could not convert date column: {e}")
    
    # Count how many countries we have
    if 'location' in df.columns:
        print(f"🌎 Number of countries/regions: {df['location'].nunique()}")
        print(f"   Top 5 countries by data points: {df['location'].value_counts().head().index.tolist()}")
    
    # Show data types and missing values summary
    print("\n📋 Data types and completeness:")
    missing_data = df.isnull().sum()
    dtypes_and_missing = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': missing_data,
        'Missing (%)': round(missing_data / len(df) * 100, 2)
    })
    print(dtypes_and_missing[dtypes_and_missing['Missing Values'] > 0])
    
    # Show a preview of the first few rows
    print("\n👀 First 5 rows of data:")
    print(df.head())
    
    return df


def clean_data(df, countries=None):
    """Clean and prepare the data for analysis"""
    print("\n🧹 Cleaning data...")
    
    # If no countries specified, use default list
    if not countries:
        countries = ['United States', 'India', 'Brazil', 'France', 'Kenya', 'South Africa']
    
    # Filter by selected countries
    filtered_df = df[df['location'].isin(countries)].copy()
    
    # Show which countries were found
    found_countries = filtered_df['location'].unique()
    print(f"📍 Selected {len(found_countries)} countries for analysis")
    
    if len(found_countries) == 0:
        print("❌ Error: None of the specified countries were found in the data!")
        print(f"Available countries include: {', '.join(df['location'].unique()[:5])}...")
        sys.exit(1)
    
    print(f"   Working with: {', '.join(found_countries)}")
    
    # Sort by country and date (important for time series)
    filtered_df = filtered_df.sort_values(['location', 'date'])
    
    # Define columns to clean
    cumulative_cols = ['total_cases', 'total_deaths', 'total_vaccinations', 
                       'people_vaccinated', 'people_fully_vaccinated']
    daily_cols = ['new_cases', 'new_deaths', 'new_vaccinations']
    
    # Handle missing values
    print("   Handling missing values...")
    
    # For cumulative columns: fill with the last known value
    for col in cumulative_cols:
        if col in filtered_df.columns:
            # Track missing values before cleaning
            missing_before = filtered_df[col].isna().sum()
            
            # Forward fill within each country (use previous day's value)
            filtered_df[col] = filtered_df.groupby('location')[col].fillna(method='ffill')
            # Any remaining NaNs at the start get 0
            filtered_df[col] = filtered_df[col].fillna(0)
            
            # Report on cleaning
            missing_after = filtered_df[col].isna().sum()
            fixed = missing_before - missing_after
            if fixed > 0:
                print(f"   - Fixed {fixed} missing values in {col}")
    
    # For daily columns: fill with 0
    for col in daily_cols:
        if col in filtered_df.columns:
            missing_before = filtered_df[col].isna().sum()
            # Fill daily values with 0
            filtered_df[col] = filtered_df[col].fillna(0)
            
            # Report on cleaning
            missing_after = filtered_df[col].isna().sum()
            fixed = missing_before - missing_after
            if fixed > 0:
                print(f"   - Fixed {fixed} missing values in {col}")
    
    # Fix negative values
    print("   Checking for negative values...")
    for col in cumulative_cols + daily_cols:
        if col in filtered_df.columns:
            # Count negative values
            neg_count = (filtered_df[col] < 0).sum()
            if neg_count > 0:
                print(f"   - Found {neg_count} negative values in {col}")
                
                # For daily columns, set negatives to 0
                if col in daily_cols:
                    filtered_df.loc[filtered_df[col] < 0, col] = 0
                    print(f"     Set negative values to 0")
                # For cumulative columns, use non-negative values only
                else:
                    for country in filtered_df['location'].unique():
                        country_mask = filtered_df['location'] == country
                        country_data = filtered_df.loc[country_mask, col]
                        
                        # Replace negative values with previous non-negative values
                        neg_mask = country_data < 0
                        neg_indices = neg_mask[neg_mask].index
                        
                        for idx in neg_indices:
                            # Find the most recent non-negative value
                            prev_values = country_data.loc[:idx].values
                            prev_non_neg = prev_values[prev_values >= 0]
                            
                            if len(prev_non_neg) > 0:
                                filtered_df.loc[idx, col] = prev_non_neg[-1]
                            else:
                                filtered_df.loc[idx, col] = 0
                    
                    print(f"     Fixed negative values using previous valid data points")
    
    # Add calculated columns that might be useful
    if all(col in filtered_df.columns for col in ['total_cases', 'total_deaths']):
        # Calculate case fatality rate (CFR)
        filtered_df['case_fatality_rate'] = np.where(
            filtered_df['total_cases'] > 0,
            filtered_df['total_deaths'] / filtered_df['total_cases'] * 100,
            np.nan
        )
    
    # Calculate 7-day rolling averages for daily metrics
    for col in daily_cols:
        if col in filtered_df.columns:
            avg_col = f"{col}_7day_avg"
            filtered_df[avg_col] = filtered_df.groupby('location')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
    
    return filtered_df


def create_visualizations(df, output_dir):
    """Create and save visualizations"""
    print("\n📈 Creating visualizations...")
    
    # Set a color palette for consistent colors across charts
    countries = df['location'].unique()
    colors = sns.color_palette("Set1", len(countries))
    country_color_dict = dict(zip(countries, colors))
    
    # CHART 1: Total Cases Over Time
    if 'total_cases' in df.columns:
        plt.figure(figsize=(12, 7))
        
        for country in countries:
            country_data = df[df['location'] == country]
            plt.plot(country_data['date'], country_data['total_cases'], 
                     label=country, color=country_color_dict[country], linewidth=2)
        
        plt.title('COVID-19 Total Cases Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Cases (log scale)', fontsize=12)
        plt.yscale('log')  # Log scale helps see all countries regardless of size
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'covid_total_cases.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved chart: {output_path}")
        plt.close()

    # CHART 2: New Cases (7-day average)
    if 'new_cases_7day_avg' in df.columns:
        plt.figure(figsize=(12, 7))
        
        for country in countries:
            country_data = df[df['location'] == country]
            plt.plot(country_data['date'], country_data['new_cases_7day_avg'], 
                     label=country, color=country_color_dict[country], linewidth=2)
        
        plt.title('COVID-19 New Cases (7-day Average)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('New Cases (7-day average)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'covid_new_cases_avg.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved chart: {output_path}")
        plt.close()

    # CHART 3: Total Deaths Over Time
    if 'total_deaths' in df.columns:
        plt.figure(figsize=(12, 7))
        
        for country in countries:
            country_data = df[df['location'] == country]
            plt.plot(country_data['date'], country_data['total_deaths'], 
                     label=country, color=country_color_dict[country], linewidth=2)
        
        plt.title('COVID-19 Total Deaths Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Deaths (log scale)', fontsize=12)
        plt.yscale('log')  # Log scale helps see all countries regardless of size
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'covid_total_deaths.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved chart: {output_path}")
        plt.close()

    # CHART 4: Case Fatality Rate
    if 'case_fatality_rate' in df.columns:
        plt.figure(figsize=(12, 7))
        
        for country in countries:
            country_data = df[df['location'] == country]
            # Create a smoother version of CFR
            country_data['cfr_smooth'] = country_data['case_fatality_rate'].rolling(window=14, min_periods=1).mean()
            plt.plot(country_data['date'], country_data['cfr_smooth'], 
                     label=country, color=country_color_dict[country], linewidth=2)
        
        plt.title('COVID-19 Case Fatality Rate Over Time (Deaths/Cases %)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Case Fatality Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'covid_cfr.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved chart: {output_path}")
        plt.close()

    # CHART 5: Vaccination Progress
    if 'people_vaccinated_per_hundred' in df.columns:
        plt.figure(figsize=(12, 7))
        
        for country in countries:
            country_data = df[df['location'] == country]
            if not country_data['people_vaccinated_per_hundred'].isna().all():
                plt.plot(country_data['date'], country_data['people_vaccinated_per_hundred'], 
                         label=country, color=country_color_dict[country], linewidth=2)
        
        plt.title('COVID-19 Vaccination Progress (% of Population)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('% of Population Vaccinated', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)  # Set y-axis from 0 to 100%
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'covid_vaccination_rate.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved chart: {output_path}")
        plt.close()
    
    # CHART 6: New Cases vs New Deaths (scatterplot)
    if all(col in df.columns for col in ['new_cases_7day_avg', 'new_deaths_7day_avg']):
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot for the most recent month of data
        recent_date = df['date'].max() - pd.Timedelta(days=30)
        recent_df = df[df['date'] >= recent_date]
        
        for country in countries:
            country_data = recent_df[recent_df['location'] == country]
            plt.scatter(
                country_data['new_cases_7day_avg'], 
                country_data['new_deaths_7day_avg'],
                label=country,
                color=country_color_dict[country],
                alpha=0.7,
                s=50
            )
        
        plt.title('COVID-19 New Cases vs New Deaths (Last 30 Days)', fontsize=16)
        plt.xlabel('New Cases (7-day avg)', fontsize=12)
        plt.ylabel('New Deaths (7-day avg)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'covid_cases_vs_deaths.png')
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved chart: {output_path}")
        plt.close()


def save_cleaned_data(df, output_dir):
    """Save the cleaned data to CSV"""
    output_file = os.path.join(output_dir, 'cleaned_covid_data.csv')
    try:
        df.to_excel(os.path.join(output_dir, 'cleaned_covid_data.xlsx'), index=False)
        print(f"\n💾 Cleaned data saved as Excel to {output_dir}/cleaned_covid_data.xlsx")
    except Exception as e:
        print(f"\n⚠️ Warning: Could not save Excel file: {e}")
        
    try:
        df.to_csv(output_file, index=False)
        print(f"💾 Cleaned data saved as CSV to {output_file}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")


def generate_insights(df):
    """Generate and print insights from the data"""
    print("\n🔍 Generating insights...")
    print("\n✨ COVID-19 DATA ANALYSIS SUMMARY ✨")
    print("=" * 60)
    
    # Overall insights
    latest_date = df['date'].max()
    earliest_date = df['date'].min()
    print(f"📊 Analysis covering data from {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
    
    # For each country, provide insights
    for country in df['location'].unique():
        country_data = df[df['location'] == country]
        # Get earliest and latest data points
        earliest = country_data.sort_values('date').iloc[0]
        latest = country_data.sort_values('date').iloc[-1]
        
        print(f"\n📍 {country}:")
        
        # Date range for this country
        if pd.notnull(earliest.get('date')) and pd.notnull(latest.get('date')):
            print(f"   Data range: {earliest['date'].strftime('%Y-%m-%d')} to {latest['date'].strftime('%Y-%m-%d')}")
        
        # Total cases
        if 'total_cases' in latest and pd.notnull(latest['total_cases']):
            print(f"   Total cases: {int(latest['total_cases']):,}")
            
            # Calculate growth rate if possible
            if 'total_cases' in earliest and pd.notnull(earliest['total_cases']) and earliest['total_cases'] > 0:
                days_diff = (latest['date'] - earliest['date']).days
                if days_diff > 0:
                    growth = latest['total_cases'] / earliest['total_cases']
                    growth_rate = (growth ** (1/days_diff) - 1) * 100
                    print(f"   Average daily growth rate: {growth_rate:.2f}% over {days_diff} days")
        
        # Total deaths
        if 'total_deaths' in latest and pd.notnull(latest['total_deaths']):
            print(f"   Total deaths: {int(latest['total_deaths']):,}")
        
        # Case fatality rate
        if 'case_fatality_rate' in latest and pd.notnull(latest['case_fatality_rate']):
            print(f"   Current case fatality rate: {latest['case_fatality_rate']:.2f}%")
        
        # Vaccination progress
        if 'people_vaccinated_per_hundred' in latest and pd.notnull(latest['people_vaccinated_per_hundred']):
            print(f"   Population vaccinated: {latest['people_vaccinated_per_hundred']:.2f}%")
        
        # Find peak cases and when they occurred
        if 'new_cases' in country_data.columns:
            max_idx = country_data['new_cases'].idxmax()
            peak_row = country_data.loc[max_idx]
            if pd.notnull(peak_row.get('date')) and pd.notnull(peak_row.get('new_cases')):
                peak_date = peak_row['date'].strftime('%Y-%m-%d')
                print(f"   Peak daily cases: {int(peak_row['new_cases']):,} on {peak_date}")
        
        # Find peak deaths and when they occurred
        if 'new_deaths' in country_data.columns:
            max_idx = country_data['new_deaths'].idxmax()
            peak_row = country_data.loc[max_idx]
            if pd.notnull(peak_row.get('date')) and pd.notnull(peak_row.get('new_deaths')):
                peak_date = peak_row['date'].strftime('%Y-%m-%d')
                print(f"   Peak daily deaths: {int(peak_row['new_deaths']):,} on {peak_date}")
        
        # Show current trends (last 14 days)
        if len(country_data) >= 14 and 'new_cases_7day_avg' in country_data.columns:
            last_14days = country_data.tail(14)
            first_week = last_14days.head(7)['new_cases_7day_avg'].mean()
            second_week = last_14days.tail(7)['new_cases_7day_avg'].mean()
            
            if second_week > first_week * 1.1:
                trend = "🔴 INCREASING"
            elif second_week < first_week * 0.9:
                trend = "🟢 DECREASING"
            else:
                trend = "🟡 STABLE"
                
            percent_change = ((second_week - first_week) / first_week) * 100 if first_week > 0 else 0
            print(f"   Recent case trend: {trend} ({percent_change:.1f}% change in last 14 days)")


def main():
    """Main function to run the COVID-19 data analysis"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='COVID-19 Data Analysis Tool')
    parser.add_argument('--file', '-f', default='owid-covid-data.csv',
                        help='Path to the COVID-19 data CSV file (default: owid-covid-data.csv)')
    parser.add_argument('--countries', '-c', nargs='+',
                        help='List of countries to analyze (default: US, India, Brazil, France, Kenya, South Africa)')
    
    args = parser.parse_args()
    
    try:
        # Setup environment
        output_dir = setup_environment()
        
        # Load data
        raw_df = load_data(args.file)
        
        # Explore data
        raw_df = explore_data(raw_df)
        
        # Clean data
        cleaned_df = clean_data(raw_df, args.countries)
        
        # Create visualizations
        create_visualizations(cleaned_df, output_dir)
        
        # Save cleaned data
        save_cleaned_data(cleaned_df, output_dir)
        
        # Generate insights
        generate_insights(cleaned_df)
        
        print(f"\n✅ Analysis complete! All charts have been saved to the {output_dir} folder.")
        print(f"📅 Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())