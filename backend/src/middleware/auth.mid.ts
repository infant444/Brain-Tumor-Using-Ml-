import { verify } from "jsonwebtoken";
import { UNAUTHARICE_STATUS } from "../constant/status.constant";

export default (req: any, res: any, next: any) => {
    const token = req.headers.access_token as string;
    // console.log(token)
    if (!token) {
        return res.status(UNAUTHARICE_STATUS).send("Login the userx")
    }

    try {
        const decoderedUser = verify(token, process.env.USER_SECRATE_KEY!);
        req.user = decoderedUser;

    } catch (e) {
        return res.status(UNAUTHARICE_STATUS).send("Login the user")
    }
    return next();
}