package com.hellblazer.art.laminar.events;

import java.io.Serializable;

public class CategoryEvent implements Serializable {
    private static final long serialVersionUID = 1L;
    private final int categoryIndex;
    private final int totalCategories;
    private final long timestamp = System.currentTimeMillis();

    public CategoryEvent(int categoryIndex, int totalCategories) {
        this.categoryIndex = categoryIndex;
        this.totalCategories = totalCategories;
    }

    public int getCategoryIndex() { return categoryIndex; }
    public int getTotalCategories() { return totalCategories; }
    public long getTimestamp() { return timestamp; }
}