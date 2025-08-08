"""
Netflix EDA - Exploratory Data Analysis
Created by: Sunil Sharma
Description: Comprehensive analysis of Netflix content library using Python, Pandas, and Matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

class NetflixEDA:
    def __init__(self, data_path=None):
        """Initialize Netflix EDA with optional data path"""
        self.data = None
        self.data_path = data_path
        self.insights = {}
        
    def load_data(self, data_path=None):
        """Load Netflix data from CSV file"""
        if data_path:
            self.data_path = data_path
            
        # If no data path provided, create sample data
        if not self.data_path:
            print("📊 No data file provided. Creating sample data for demonstration...")
            self.create_sample_data()
            return True
            
        try:
            # Try to load the data
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
            print(f"📊 Columns: {list(self.data.columns)}")
            return True
        except FileNotFoundError:
            print("❌ Data file not found. Creating sample data for demonstration...")
            self.create_sample_data()
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample Netflix data for demonstration"""
        np.random.seed(42)
        
        # Generate sample data
        n_samples = 5000
        
        # Years (2010-2023)
        years = np.random.choice(range(2010, 2024), n_samples)
        
        # Content types
        content_types = np.random.choice(['Movie', 'TV Show'], n_samples, p=[0.6, 0.4])
        
        # Genres
        movie_genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Horror', 
                       'Documentary', 'Animation', 'Adventure', 'Sci-Fi']
        tv_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Documentary', 'Reality TV',
                    'Crime', 'Romance', 'Horror', 'Animation']
        
        genres = []
        for content_type in content_types:
            if content_type == 'Movie':
                genres.append(np.random.choice(movie_genres))
            else:
                genres.append(np.random.choice(tv_genres))
        
        # Ratings
        ratings = np.random.choice(['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'PG', 'TV-Y', 'TV-Y7'], n_samples)
        
        # Duration
        durations = []
        for content_type in content_types:
            if content_type == 'Movie':
                durations.append(f"{np.random.randint(80, 180)} min")
            else:
                seasons = np.random.randint(1, 8)
                episodes = np.random.randint(8, 24)
                durations.append(f"{seasons} Season{'s' if seasons > 1 else ''}")
        
        # Countries
        countries = np.random.choice(['United States', 'India', 'United Kingdom', 'Canada', 
                                   'South Korea', 'Japan', 'Spain', 'France', 'Germany', 'Brazil'], n_samples)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'show_id': [f's{i:04d}' for i in range(1, n_samples + 1)],
            'type': content_types,
            'title': [f'Title {i}' for i in range(1, n_samples + 1)],
            'director': [f'Director {i}' for i in range(1, n_samples + 1)],
            'cast': [f'Actor {i}, Actor {i+1}' for i in range(1, n_samples + 1)],
            'country': countries,
            'date_added': [f"{np.random.randint(1, 13):02d}/{np.random.randint(1, 29):02d}/{year}" 
                          for year in years],
            'release_year': years,
            'rating': ratings,
            'duration': durations,
            'listed_in': genres,
            'description': [f'Description for title {i}' for i in range(1, n_samples + 1)]
        })
        
        print("✅ Sample data created successfully!")
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n" + "="*50)
        print("📊 NETFLIX DATA EXPLORATION")
        print("="*50)
        
        print(f"\n📈 Dataset Shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")
        
        print("\n📊 Data Types:")
        print(self.data.dtypes)
        
        print("\n🔍 Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\n📈 Basic Statistics:")
        print(self.data.describe())
        
        return self.data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n🧹 CLEANING DATA...")
        
        # Convert date_added to datetime
        self.data['date_added'] = pd.to_datetime(self.data['date_added'], errors='coerce')
        
        # Extract year from date_added
        self.data['year_added'] = self.data['date_added'].dt.year
        
        # Clean duration column
        self.data['duration_clean'] = self.data['duration'].str.extract(r'(\d+)').astype(float)
        
        # Split genres for analysis
        self.data['genres_list'] = self.data['listed_in'].str.split(', ')
        
        print("✅ Data cleaning completed!")
        return self.data
    
    def analyze_content_distribution(self):
        """Analyze content type distribution"""
        print("\n📺 CONTENT TYPE ANALYSIS")
        print("-" * 30)
        
        content_counts = self.data['type'].value_counts()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        ax1.pie(content_counts.values, labels=content_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['#E50914', '#564D4D'])
        ax1.set_title('Content Type Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        content_counts.plot(kind='bar', ax=ax2, color=['#E50914', '#564D4D'])
        ax2.set_title('Content Type Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('plots/content_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: plots/content_distribution.png")
        
        # Store insights
        self.insights['content_distribution'] = {
            'movies_percentage': (content_counts['Movie'] / content_counts.sum()) * 100,
            'tv_shows_percentage': (content_counts['TV Show'] / content_counts.sum()) * 100
        }
        
        print(f"📊 Movies: {content_counts['Movie']} ({self.insights['content_distribution']['movies_percentage']:.1f}%)")
        print(f"📺 TV Shows: {content_counts['TV Show']} ({self.insights['content_distribution']['tv_shows_percentage']:.1f}%)")
    
    def analyze_genre_trends(self):
        """Analyze genre trends and popularity"""
        print("\n🎭 GENRE ANALYSIS")
        print("-" * 30)
        
        # Explode genres for analysis
        genre_df = self.data.explode('genres_list')
        genre_counts = genre_df['genres_list'].value_counts().head(15)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Top genres bar chart
        genre_counts.plot(kind='barh', ax=ax1, color='#E50914')
        ax1.set_title('Top 15 Genres on Netflix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Count')
        
        # Genre by content type
        genre_by_type = genre_df.groupby(['genres_list', 'type']).size().unstack(fill_value=0)
        genre_by_type.head(10).plot(kind='bar', ax=ax2, color=['#E50914', '#564D4D'])
        ax2.set_title('Genre Distribution by Content Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.legend(title='Content Type')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: plots/genre_analysis.png")
        
        # Store insights
        self.insights['top_genres'] = genre_counts.head(5).to_dict()
        
        print("🏆 Top 5 Genres:")
        for i, (genre, count) in enumerate(genre_counts.head(5).items(), 1):
            print(f"{i}. {genre}: {count}")
    
    def analyze_ratings_distribution(self):
        """Analyze content ratings distribution"""
        print("\n📋 RATINGS ANALYSIS")
        print("-" * 30)
        
        rating_counts = self.data['rating'].value_counts()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        ax1.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%', 
                startangle=90)
        ax1.set_title('Content Ratings Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        rating_counts.plot(kind='bar', ax=ax2, color='#E50914')
        ax2.set_title('Content Ratings Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/ratings_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: plots/ratings_distribution.png")
        
        # Store insights
        self.insights['ratings_distribution'] = rating_counts.to_dict()
        
        print("📊 Ratings Distribution:")
        for rating, count in rating_counts.items():
            percentage = (count / rating_counts.sum()) * 100
            print(f"  {rating}: {count} ({percentage:.1f}%)")
    
    def analyze_yearly_growth(self):
        """Analyze content growth over years"""
        print("\n📈 YEARLY GROWTH ANALYSIS")
        print("-" * 30)
        
        # Yearly content addition
        yearly_content = self.data['release_year'].value_counts().sort_index()
        yearly_added = self.data['year_added'].value_counts().sort_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Release year trend
        yearly_content.plot(kind='line', marker='o', ax=ax1, color='#E50914', linewidth=2)
        ax1.set_title('Content Release Year Trend', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Titles')
        ax1.grid(True, alpha=0.3)
        
        # Year added trend
        yearly_added.plot(kind='line', marker='s', ax=ax2, color='#564D4D', linewidth=2)
        ax2.set_title('Content Added to Netflix by Year', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Titles Added')
        ax2.set_xlabel('Year')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/yearly_growth.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: plots/yearly_growth.png")
        
        # Store insights
        self.insights['yearly_growth'] = {
            'peak_release_year': yearly_content.idxmax(),
            'peak_added_year': yearly_added.idxmax(),
            'total_released': yearly_content.sum(),
            'total_added': yearly_added.sum()
        }
        
        print(f"📊 Peak Release Year: {self.insights['yearly_growth']['peak_release_year']}")
        print(f"📊 Peak Added Year: {self.insights['yearly_growth']['peak_added_year']}")
        print(f"📊 Total Titles Released: {self.insights['yearly_growth']['total_released']}")
        print(f"📊 Total Titles Added: {self.insights['yearly_growth']['total_added']}")
    
    def analyze_country_distribution(self):
        """Analyze content by country"""
        print("\n🌍 COUNTRY ANALYSIS")
        print("-" * 30)
        
        # Clean country data and get top countries
        country_counts = self.data['country'].value_counts().head(15)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        country_counts.plot(kind='barh', color='#E50914')
        ax.set_title('Top 15 Countries by Content Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        
        plt.tight_layout()
        plt.savefig('plots/country_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: plots/country_distribution.png")
        
        # Store insights
        self.insights['top_countries'] = country_counts.head(10).to_dict()
        
        print("🌍 Top 10 Countries:")
        for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
            print(f"{i}. {country}: {count}")
    
    def analyze_duration_trends(self):
        """Analyze duration trends for movies"""
        print("\n⏱️ DURATION ANALYSIS")
        print("-" * 30)
        
        # Filter movies only
        movies = self.data[self.data['type'] == 'Movie'].copy()
        movies['duration_min'] = movies['duration_clean']
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Duration distribution
        movies['duration_min'].hist(bins=30, ax=ax1, color='#E50914', alpha=0.7)
        ax1.set_title('Movie Duration Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_ylabel('Frequency')
        
        # Duration by genre
        top_genres = movies['listed_in'].value_counts().head(8).index
        genre_duration = movies[movies['listed_in'].isin(top_genres)].groupby('listed_in')['duration_min'].mean()
        genre_duration.plot(kind='bar', ax=ax2, color='#564D4D')
        ax2.set_title('Average Duration by Genre', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Duration (minutes)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: plots/duration_analysis.png")
        
        # Store insights
        self.insights['duration_analysis'] = {
            'avg_movie_duration': movies['duration_min'].mean(),
            'median_movie_duration': movies['duration_min'].median(),
            'longest_genre': genre_duration.idxmax(),
            'shortest_genre': genre_duration.idxmin()
        }
        
        print(f"📊 Average Movie Duration: {self.insights['duration_analysis']['avg_movie_duration']:.1f} minutes")
        print(f"📊 Median Movie Duration: {self.insights['duration_analysis']['median_movie_duration']:.1f} minutes")
        print(f"📊 Longest Genre: {self.insights['duration_analysis']['longest_genre']}")
        print(f"📊 Shortest Genre: {self.insights['duration_analysis']['shortest_genre']}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("📊 NETFLIX EDA SUMMARY REPORT")
        print("="*60)
        print("👨‍💻 Created by: Sunil Sharma")
        print("="*60)
        
        print(f"\n📈 Dataset Overview:")
        print(f"  • Total Titles: {len(self.data):,}")
        print(f"  • Movies: {len(self.data[self.data['type'] == 'Movie']):,}")
        print(f"  • TV Shows: {len(self.data[self.data['type'] == 'TV Show']):,}")
        
        print(f"\n🎭 Content Analysis:")
        print(f"  • Top Genre: {list(self.insights.get('top_genres', {}).keys())[0] if self.insights.get('top_genres') else 'N/A'}")
        print(f"  • Most Popular Rating: {list(self.insights.get('ratings_distribution', {}).keys())[0] if self.insights.get('ratings_distribution') else 'N/A'}")
        print(f"  • Top Country: {list(self.insights.get('top_countries', {}).keys())[0] if self.insights.get('top_countries') else 'N/A'}")
        
        print(f"\n📈 Growth Trends:")
        if self.insights.get('yearly_growth'):
            print(f"  • Peak Release Year: {self.insights['yearly_growth']['peak_release_year']}")
            print(f"  • Peak Added Year: {self.insights['yearly_growth']['peak_added_year']}")
        
        print(f"\n⏱️ Duration Insights:")
        if self.insights.get('duration_analysis'):
            print(f"  • Average Movie Duration: {self.insights['duration_analysis']['avg_movie_duration']:.1f} minutes")
            print(f"  • Longest Genre: {self.insights['duration_analysis']['longest_genre']}")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis"""
        print("🚀 Starting Netflix EDA Analysis...")
        
        # Load and explore data
        if self.load_data():
            self.explore_data()
            self.clean_data()
        else:
            print("❌ Failed to load data. Exiting...")
            return None
        
        # Run all analyses
        self.analyze_content_distribution()
        self.analyze_genre_trends()
        self.analyze_ratings_distribution()
        self.analyze_yearly_growth()
        self.analyze_country_distribution()
        self.analyze_duration_trends()
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n✅ Netflix EDA Analysis Complete!")
        print("📁 All plots saved in the 'plots' directory!")
        return self.insights

# Main execution
if __name__ == "__main__":
    # Initialize and run analysis
    netflix_eda = NetflixEDA()
    insights = netflix_eda.run_complete_analysis()
