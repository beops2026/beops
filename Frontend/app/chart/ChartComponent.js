import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { LoadingSpinner } from "./components/shared/LoadingSpinner";
import { ErrorDisplay } from "./components/shared/ErrorDisplay";
import { DateControls } from "./components/controls/DateControls";
import { TimeNavigation } from "./components/controls/TimeNavigation";
import { ViewControls } from "./components/controls/ViewControls";
import { LoadChart } from "./components/chart/LoadChart";
import { TableView } from "./components/tables/TableView";
import { LoadStats } from "./components/stats/LoadStats";
import { Button } from "@/components/ui/button";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 100, damping: 12 }
  }
};

// Helper function to calculate average of array of numbers
const calculateAverage = (numbers) => {
  if (!numbers || numbers.length === 0) return 0;
  return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;
};

// Helper function to format date for display
const formatDateForDisplay = (date) => {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric'
  });
};

export default function ChartComponent({ 
  startDate, 
  endDate, 
  viewType = '5min',
  onViewChange 
}) {
  const [data, setData] = useState([]);
  const [previousData, setPreviousData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(startDate);
  const [showTable, setShowTable] = useState(false);
  const [dateMode, setDateMode] = useState('single');
  const [currentViewType, setCurrentViewType] = useState(viewType);

  // Update internal view type when prop changes
  useEffect(() => {
    setCurrentViewType(viewType);
  }, [viewType]);

  const formatDateForBackend = (date) => {
    const adjustedDate = new Date(date);
    
    const year = adjustedDate.getFullYear();
    const month = String(adjustedDate.getMonth() + 1).padStart(2, '0');
    const day = String(adjustedDate.getDate()).padStart(2, '0');
    const hours = String(adjustedDate.getHours()).padStart(2, '0');
    const minutes = String(adjustedDate.getMinutes()).padStart(2, '0');
    const seconds = String(adjustedDate.getSeconds()).padStart(2, '0');

    return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
  };

  // Function to aggregate data based on view type
  const aggregateData = (rawData, viewType) => {
    if (!rawData || rawData.length === 0) return [];

    switch (viewType) {
      case '5min':
        return rawData.map(item => ({
          time: new Date(item.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
          }),
          load: parseFloat(item.value),
          timestamp: item.timestamp
        }));

      case 'hourly': {
        // Group data by hour and calculate averages
        const hourlyGroups = {};
        rawData.forEach(item => {
          const date = new Date(item.timestamp);
          const hourKey = date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
          }).slice(0, 2) + ':00';
          
          if (!hourlyGroups[hourKey]) {
            hourlyGroups[hourKey] = [];
          }
          hourlyGroups[hourKey].push(parseFloat(item.value));
        });

        return Object.entries(hourlyGroups).map(([hour, values]) => ({
          time: hour,
          load: calculateAverage(values),
          timestamp: new Date(rawData[0].timestamp).setHours(parseInt(hour))
        })).sort((a, b) => a.timestamp - b.timestamp);
      }

      case 'weekly': {
        // Group data by day and hour for the week
        const dailyGroups = {};
        rawData.forEach(item => {
          const date = new Date(item.timestamp);
          const dayKey = date.toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric'
          });
          
          if (!dailyGroups[dayKey]) {
            dailyGroups[dayKey] = [];
          }
          dailyGroups[dayKey].push(parseFloat(item.value));
        });

        return Object.entries(dailyGroups).map(([day, values]) => ({
          time: day,
          load: calculateAverage(values),
          timestamp: new Date(day).getTime()
        })).sort((a, b) => a.timestamp - b.timestamp);
      }

      case 'monthly': {
        // Group data by day and calculate averages
        const dailyGroups = {};
        rawData.forEach(item => {
          const date = new Date(item.timestamp);
          const dateKey = formatDateForDisplay(date);
          
          if (!dailyGroups[dateKey]) {
            dailyGroups[dateKey] = [];
          }
          dailyGroups[dateKey].push(parseFloat(item.value));
        });

        return Object.entries(dailyGroups).map(([date, values]) => ({
          time: date,
          load: calculateAverage(values),
          timestamp: new Date(date).getTime()
        })).sort((a, b) => a.timestamp - b.timestamp);
      }

      default:
        return rawData;
    }
  };

  const fetchPredictions = async (start, end) => {
    try {
      const startStr = formatDateForBackend(start);
      const endStr = formatDateForBackend(end);

      console.log('Sending request with dates:', {
        startDateTime: startStr,
        endDateTime: endStr
      });

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          startDateTime: startStr,
          endDateTime: endStr
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', errorText);
        throw new Error(`API request failed: ${errorText}`);
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Failed to get predictions');
      }

      // Aggregate the data based on view type
      return aggregateData(result.predictions, viewType);
    } catch (error) {
      console.error('Error fetching predictions:', error);
      throw error;
    }
  };

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      let start, end;
      
      if (dateMode === 'range') {
        // In range mode, use the start and end dates directly
        start = new Date(startDate);
        end = new Date(endDate);
      } else {
        // In single date mode, calculate based on view type
        const referenceDate = new Date(selectedDate);
        
        switch (viewType) {
          case '5min':
          case 'hourly':
            start = new Date(referenceDate);
            start.setHours(0, 0, 0, 0);
            end = new Date(referenceDate);
            end.setHours(23, 59, 59, 999);
            break;
            
          case 'weekly':
            // Start from the selected date
            start = new Date(referenceDate);
            start.setHours(0, 0, 0, 0);
            // End after 7 days
            end = new Date(referenceDate);
            end.setDate(end.getDate() + 6);
            end.setHours(23, 59, 59, 999);
            break;
            
          case 'monthly':
            start = new Date(referenceDate.getFullYear(), referenceDate.getMonth(), 1, 0, 0, 0);
            end = new Date(referenceDate.getFullYear(), referenceDate.getMonth() + 1, 0, 23, 59, 59);
            break;
            
          default:
            throw new Error('Invalid view type');
        }
      }

      // Fetch predicted data
      const currentData = await fetchPredictions(start, end);

      // Fetch actual data from Delhi SLDC API with appropriate interval
      const actualDataResponse = await fetch(`/api/load-data?date=${formatDateForBackend(start)}&interval=${dateMode === 'range' ? '5min' : viewType}`);
      const actualData = await actualDataResponse.json();

      // Combine predicted and actual data
      const combinedData = currentData.map(item => {
        const actualDataPoint = actualData.find(actual => actual.time === item.time);
        return {
          ...item,
          actualLoad: actualDataPoint ? actualDataPoint.load : null
        };
      });

      setData(combinedData);

      // Only fetch previous data in single date mode
      if (dateMode === 'single') {
        const prevStart = new Date(start);
        const prevEnd = new Date(end);
        
        switch (viewType) {
          case '5min':
          case 'hourly':
            prevStart.setDate(prevStart.getDate() - 1);
            prevEnd.setDate(prevEnd.getDate() - 1);
            break;
          case 'weekly':
            prevStart.setDate(prevStart.getDate() - 7);
            prevEnd.setDate(prevEnd.getDate() - 7);
            break;
          case 'monthly':
            prevStart.setMonth(prevStart.getMonth() - 1);
            prevEnd.setMonth(prevEnd.getMonth() - 1);
            break;
        }

        const previousPeriodData = await fetchPredictions(prevStart, prevEnd);
        setPreviousData(previousPeriodData);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    setSelectedDate(startDate);
  }, [startDate]);

  useEffect(() => {
    if (selectedDate) {
      fetchData();
    }
  }, [selectedDate, viewType]);

  const handleTimeChange = (hours) => {
    const newDate = new Date(selectedDate);
    newDate.setHours(newDate.getHours() + hours);
    setSelectedDate(newDate);
  };

  const handleViewTypeChange = (newViewType) => {
    // Simply change the view type without modifying the date
    onViewChange(newViewType);
  };

  const handleDateChange = (days) => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() + days);
    setSelectedDate(newDate);
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorDisplay error={error} onRetry={() => fetchData()} />;
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="space-y-4 text-white"
    >
      {/* Date Mode Controls */}
      <div className="flex gap-2 mb-4">
        <Button
          onClick={() => setDateMode('single')}
          className={`${dateMode === 'single' ? 'bg-blue-500' : 'bg-blue-500/20'} hover:bg-blue-500/30 text-white`}
        >
          Single Date
        </Button>
        <Button
          onClick={() => setDateMode('range')}
          className={`${dateMode === 'range' ? 'bg-blue-500' : 'bg-blue-500/20'} hover:bg-blue-500/30 text-white`}
        >
          Date Range
        </Button>
      </div>

      {dateMode === 'single' ? (
        <>
          <DateControls 
            selectedDate={selectedDate} 
            onDateChange={setSelectedDate}
            onPrevious={() => handleDateChange(-1)}
            onNext={() => handleDateChange(1)}
          />
          <ViewControls 
            viewType={viewType}
            onViewChange={onViewChange}
            showTable={showTable}
            onToggleTable={() => setShowTable(!showTable)}
          />
        </>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="text-sm text-gray-400">
            Showing 5-minute data for selected date range
          </div>
          {/* Your existing date range picker component */}
        </div>
      )}

      <LoadStats 
        data={data}
        previousData={dateMode === 'single' ? previousData : []}
        viewType={dateMode === 'range' ? '5min' : viewType}
      />

      {showTable ? (
        <TableView 
          viewType={dateMode === 'range' ? '5min' : viewType} 
          data={data}
        />
      ) : (
        <LoadChart 
          data={data} 
          viewType={dateMode === 'range' ? '5min' : viewType} 
          itemVariants={itemVariants}
        />
      )}
    </motion.div>
  );
}