import { Schema, Types, model } from "mongoose";

export interface Contect {
    name: string;
    email: string;
    message: string;
    date: string;
    response:string;
}

export const ContectSchema = new Schema<Contect>(
    {
        name: { type: String, required: true },
        email: { type: String, required: true },
        message: { type: String, required: true },
        date: { type: String, required: true },
        response:{type:String}
    }, {
    toJSON: {
        virtuals: true
    },
    toObject: {
        virtuals: true
    },
    timestamps: true

}
);

export const ContectModel = model<Contect>('feedback_quary', ContectSchema);
