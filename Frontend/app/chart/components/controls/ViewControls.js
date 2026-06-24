import { Button } from "@/components/ui/button";

export function ViewControls({ viewType, onViewChange, showTable, onToggleTable, disabled = false }) {
  return (
    <div className="flex justify-between items-center">
      <div className="flex gap-2">
        <Button
          onClick={() => !disabled && onViewChange('5min')}
          className={`${viewType === '5min' ? 'bg-blue-500' : 'bg-blue-500/20'} 
            hover:bg-blue-500/30 text-white
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={disabled}
        >
          5 Min
        </Button>
        <Button
          onClick={() => !disabled && onViewChange('hourly')}
          className={`${viewType === 'hourly' ? 'bg-blue-500' : 'bg-blue-500/20'} 
            hover:bg-blue-500/30 text-white
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={disabled}
        >
          Hourly
        </Button>
        <Button
          onClick={() => !disabled && onViewChange('weekly')}
          className={`${viewType === 'weekly' ? 'bg-blue-500' : 'bg-blue-500/20'} 
            hover:bg-blue-500/30 text-white
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={disabled}
        >
          Weekly
        </Button>
        <Button
          onClick={() => !disabled && onViewChange('monthly')}
          className={`${viewType === 'monthly' ? 'bg-blue-500' : 'bg-blue-500/20'} 
            hover:bg-blue-500/30 text-white
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={disabled}
        >
          Monthly
        </Button>
      </div>

      <Button
        onClick={onToggleTable}
        className={`bg-blue-500/20 hover:bg-blue-500/30 text-white
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        disabled={disabled}
      >
        {showTable ? 'Show Chart' : 'Show Table'}
      </Button>
    </div>
  );
} 