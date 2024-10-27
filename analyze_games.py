import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def analyze_games_data(file_path):
    """
    Analyze video games sales data to uncover interesting trends and patterns
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Clean the data
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')

    # 1. Calculate success rate by genre
    genre_success = df.groupby('Genre').agg({
        'Global_Sales': ['count', 'sum', 'mean'],
        'Critic_Score': 'mean'
    }).round(2)

    # 2. Analyze regional preferences
    df['NA_Ratio'] = df['NA_Sales'] / df['Global_Sales']
    df['EU_Ratio'] = df['EU_Sales'] / df['Global_Sales']
    df['JP_Ratio'] = df['JP_Sales'] / df['Global_Sales']

    regional_preferences = df.groupby('Genre').agg({
        'NA_Ratio': 'mean',
        'EU_Ratio': 'mean',
        'JP_Ratio': 'mean'
    }).round(2)

    # 3. Publisher performance over time
    top_publishers = df.groupby('Publisher')['Global_Sales'].sum().nlargest(10).index
    publisher_timeline = df[df['Publisher'].isin(top_publishers)].groupby(['Publisher', 'Year_of_Release'])[
        'Global_Sales'].sum().reset_index()

    # 4. Platform lifecycle analysis
    platform_lifecycle = df.groupby(['Platform', 'Year_of_Release']).agg({
        'Global_Sales': 'sum',
        'Name': 'count'
    }).reset_index()

    # 5. Critical reception vs commercial success
    correlation = df['Critic_Score'].corr(df['Global_Sales'])

    # Generate visualizations
    plt.figure(figsize=(15, 10))

    # Plot 1: Genre Success
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='Genre', y='Global_Sales', estimator='sum')
    plt.xticks(rotation=45)
    plt.title('Total Sales by Genre')

    # Plot 2: Regional Preferences Heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(regional_preferences, cmap='YlOrRd', annot=True)
    plt.title('Regional Market Preferences by Genre')

    # Plot 3: Publisher Timeline
    plt.subplot(2, 2, 3)
    for publisher in top_publishers[:5]:  # Top 5 for clarity
        publisher_data = publisher_timeline[publisher_timeline['Publisher'] == publisher]
        plt.plot(publisher_data['Year_of_Release'], publisher_data['Global_Sales'], label=publisher)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('Top Publishers Performance Over Time')
    plt.xticks(rotation=45)

    # Plot 4: Critic Score vs Sales
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='Critic_Score', y='Global_Sales', alpha=0.5)
    plt.title(f'Critic Score vs Sales (correlation: {correlation:.2f})')

    plt.tight_layout()
    plt.show(block=False)

    # Return key insights
    insights = {
        'top_genres': genre_success.sort_values(('Global_Sales', 'sum'), ascending=False).head(),
        'regional_preferences': regional_preferences,
        'critical_commercial_correlation': correlation,
        'top_publishers': df.groupby('Publisher')['Global_Sales'].sum().nlargest(5)
    }

    return insights


# Example usage
if __name__ == "__main__":
    insights = analyze_games_data('video_games.csv')
    print("\nKey Insights:")
    print("=============")
    print("\nTop Performing Genres:")
    print(insights['top_genres'])
    print("\nRegional Market Preferences:")
    print(insights['regional_preferences'])
    print(f"\nCorrelation between Critic Scores and Sales: {insights['critical_commercial_correlation']:.2f}")
    print("\nTop Publishers by Global Sales:")
    print(insights['top_publishers'])

    # Keep the script running until user presses Enter
    input("\nPress Enter to close the plots and exit...")
